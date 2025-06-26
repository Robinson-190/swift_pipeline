"""
This assumes the entire observation has been downloaded into a subdirectory
Subdirectory should be named after the observation ID
CALDB should be initiated before python is invoked
"""
import warnings
import os
import numpy as np
from pathlib import Path
import fnmatch
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import re
from astropy.io import fits
import astropy.units as u
import astropy.wcs as wcs
from astropy.coordinates import SkyCoord
from regions import Regions, RectangleSkyRegion

try:
    from heasoftpy import xrtpipeline,barycorr,xrtproducts
except ImportError:
    warnings.warn("\nWARNING:Could not import 'heasoftpy'. HEASOFT may not be initialised in this window.")

src_col='navy'
bkg_col='red'
fontsize_legend=5

if 'CALDBALIAS' not in os.popen('env').read():
    warnings.warn("\nWARNING:'CALDBALIAS' not found in environmental variables.\
                  \nCALDB may not be initialised in this window.")
                  
if 'LHEASOFT' not in os.popen('env').read():
    warnings.warn("\nWARNING:'LHEASOFT' not found in environmental variables.\
                  \nHEASOFT may not be initialised in this window.")


def detect_modes(data_folder: str):
    """
    Check, from original events folder, how many modes were actively used
    durng the pointing of the telescope.

    Parameters
    ----------
    folder : str
        Folder to check.

    Returns
    -------
    List containing tuples of mode and window number.

    """

    modes = []

    if data_folder[-1] != "/":
        data_folder += "/"
        
    folder=data_folder#+'xrt/event/'
    all_combinations = []
    for file in os.listdir(folder):
        if len(file)>17:
            if file[17] in "12345":  # If window is a number
                all_combinations.append((file[14:16], file[17]))
    
    unique_combnations = list(set(all_combinations))
    
    if not unique_combnations:
        raise Exception(
            f"Modes and window numbers could not be automatically parsed from event file names in \
                {folder}"
        )
    modes=unique_combnations
    return modes


def bool_cmd_str(boolean: bool):
    """Converts boolean to "yes" or "no" for use in the command line arguments

    Args:
        bool (bool): boolean to convert

    Returns:
        str: "yes" or "no"
    """
    return "yes" if boolean else "no"


def obsid_from_folder(folder_name: str):
    """Attempt to extract the observation ID from the folder name

    Args:
        folder_name (str): name of the folder. If downloaded using `swifttools`
        then the folder name will be the observation ID

    Returns:
        str: Assumed observation ID from foldername
    """
    return (folder_name.split("/"))[-2]


def source_region(filename):
    with open(filename, "r") as file:
        contents = file.read()

    return contents


def circle_region_parameters(circular_region_str):
    circle_search = re.search('\((.*)\)',circular_region_str, re.IGNORECASE)
    x, y, r = circle_search.group(1).split(",")

    return float(x), float(y), float(r)


def annulus_region_parameters_from_circle(x, y, r, gap_ratio=1.2, area_ratio=10.0):
    r_inner = int(np.ceil(gap_ratio * float(r)))

    r_outer = int(np.ceil(np.sqrt(r**2 * (area_ratio + gap_ratio**2))))

    return float(x), float(y), float(r_inner), float(r_outer)


def annulus_from_circle(circle_region, gap_ratio=1.2, area_ratio=10.0):
    """Returns an annulus region around a circle for use as a background region
    in DS9

    Args:
        circle_region (str): DS9 region of the source
        gap_ratio (float, optional): Ratio of the inner annulus circle to the
        source region. Defaults to 1.2.
        area_ratio (float, optional): Ratio of the desired annulus area to the
        are of the circle. Defaults to 10.0.

    Returns:
        str: DS9 string on annulus region
    """
    x, y, r = circle_region_parameters(circle_region)
    r = float(r)

    x, y, r_inner, r_outer = annulus_region_parameters_from_circle(x, y, r, gap_ratio, area_ratio)

    return f"IMAGE;ANNULUS({x},{y},{r_inner},{r_outer})"

def set_image_coordinate_region(filename):
    
    coordinate_systems=['PHYSICAL', 'IMAGE', 'FK4', 'B1950', 'FK5', 'J2000', 'ICRS', 'GALACTIC', 'ECLIPTIC', 'WCS', 'WCSA', 'LINEAR']
    
    with open(filename, "r") as file:
        contents = file.read()
    
    has_coordinate_system=False
    for coordinate_system in coordinate_systems:
        if coordinate_system in contents:
            has_coordinate_system=True
            break
        
        
    if not has_coordinate_system:
        with open(filename, "w") as file:
            file.write(f"IMAGE;{contents}")


def write_annulus_region(source_region_file: str, background_region_file: str = "", gap_ratio=1.2, area_ratio=10.0):
    circle_str = source_region(source_region_file)
    if 'circle' in circle_str.lower():#Do not rewrite annulus files
        set_image_coordinate_region(source_region_file)
        annulus_string = annulus_from_circle(circle_str, gap_ratio=1.2, area_ratio=10.0)
        if not background_region_file:
            background_region_file = source_region_file.replace(".reg", "_automatic_background.reg")
        with open(background_region_file, "w") as file:
            file.write(annulus_string)
        
        
def automatic_region_generation(folder, gap_ratio=1.2, area_ratio=10.0):
    if folder[-1]!='/':folder+='/'
    region_files = fnmatch.filter(os.listdir(folder),'*.reg*')
    
    for region_file in region_files:
        write_annulus_region(source_region_file=folder+region_file, gap_ratio=gap_ratio, area_ratio=area_ratio)


def observation_details_from_pat(folder: str):
    """Retrieve a dictionary containing various details of the observation from
    the Primary Attitude FITS  file
    Args:
        folder (str): Folder in which the Primary Attitude FITS  file is stored
        obs_id (str): observation ID
    Returns:
        dict: Returns dictionary containing bgfilevarious details of the
        observation
    """
    obs_details_dict = {}
    if folder[-1] != "/":
        folder += "/"

    if "auxil" not in folder:
        auxil_folder = folder + "auxil/"
    else:
        auxil_folder = folder

    pat_file = [filename for filename in os.listdir(auxil_folder) if "pat.fits" in filename][0]

    with fits.open(auxil_folder + pat_file) as hdul:
        obs_details_dict["object_ra"] = hdul[0].header["RA_OBJ"]
        obs_details_dict["object_dec"] = hdul[0].header["DEC_OBJ"]
        obs_details_dict["start time"] = hdul[0].header["DATE-OBS"]
        obs_details_dict["end time"] = hdul[0].header["DATE-END"]
        obs_details_dict["OBS_ID"] = hdul[0].header["OBS_ID"]

    return obs_details_dict

def run_xrtpipeline(
    raw_data_folder: str,
    output_folder: str,
    obs_id: str = "",
    ra_decimal: float = -np.inf,
    dec_decimal: float = -np.inf,
    dry_run=False,
    cleanup=False,
    clobber=True,
    createexpomap=True,
):
    """Produces cleaned event files, using the XRT-PIPELINE.
    Applies barycentric corrections to to the event, housekeeping and attitude
    files

    Args:
        raw_data_folder (str): the folder in which the data is stored
        obs_id (str): the observation ID of the SwiftXRT observation
        ra_decimal (float): the right-ascension of the source, in decimal
        degrees
        dec_decimal (float): the declination of the source, in decimal degrees
    """

    if Path(output_folder).is_relative_to(Path(raw_data_folder)):
        raise Exception(
            "The output directory cannot be a subdirectory of the input\
            directory."
        )

    if raw_data_folder[-1] != "/":
        raw_data_folder += "/"

    if output_folder[-1] != "/":
        output_folder += "/"

    if os.path.exists(output_folder) is False and dry_run is False:
        os.makedirs(output_folder)

    cleanup = bool_cmd_str(cleanup)
    clobber = bool_cmd_str(clobber)
    createexpomap = bool_cmd_str(createexpomap)

    obs_details_dict = observation_details_from_pat(folder=raw_data_folder)
    obs_id = obs_details_dict["OBS_ID"]

    if obs_details_dict["OBS_ID"] != obs_id:
        print(
            f"WARNING: the observation ID given dopes not correspond to the\
                PAT ({obs_details_dict['OBS_ID']},{obs_id})"
        )

    if ra_decimal == -np.inf:
        ra_decimal = obs_details_dict["object_ra"]
    if dec_decimal == -np.inf:
        dec_decimal = obs_details_dict["object_dec"]
    
    A=xrtpipeline(indir=raw_data_folder,outdir=output_folder,
                  steminputs=f'sw{obs_id}',
                  srcra=str(ra_decimal) ,srcdec=str(dec_decimal),
                  cleanup=cleanup, clobber=clobber,
                  createexpomap=createexpomap,
                  noprompt=True,allow_failure=True)

    with open(f'{output_folder}/log_{obs_id}','w') as file:
        file.write(str(A))


def run_barycentric_correction(
    data_folder: str,
    xrtpipeline_output_folder: str,
    obs_id: str = "",
    clobber: bool = True,
    mode: str = None,
    window: str = None,
):
    """Run the commands to do a barycentric correction on the data

    Args:
        data_folder (str): Folder in which the data is contained
        xrtpipeline_output_folder (str): The folder in which the output from
        the xrtpipeline is stored
        obs_id (str, optional): Observation ID Defaults to "".
        dry_run (bool, optional): If `dry_run==True` then the commands are only
        printed. If `dry_run==False` then the commands are executed. Defaults
        to True.
        clobber (bool, optional): Can files be overwritten if necessary.
        Defaults to True.
    """

    obs_details_dict = observation_details_from_pat(folder=data_folder)
    obs_id = obs_details_dict["OBS_ID"]

    if len(obs_id) == 0:
        obs_id = obs_details_dict["OBS_ID"]

    if xrtpipeline_output_folder[-1] != "/":
        xrtpipeline_output_folder += "/"
    if data_folder[-1] != "/":
        data_folder += "/"

    copy_attitude_file_command = f"cp {data_folder}auxil/sw{obs_id}pat.fits.gz\
 {xrtpipeline_output_folder}"
    copy_ephemeris_file_command = f"cp \
{data_folder}auxil/sw{obs_id}sao.fits.gz \
{xrtpipeline_output_folder}"

    os.system(copy_attitude_file_command)
    os.system(copy_ephemeris_file_command)

    

    if not mode:
        modes_list = detect_modes(data_folder+'/xrt/event/')
    else:
        modes_list = [(mode, window)]

    for mode, window in modes_list:
        
        files_for_barycentric_correction = [
            f"x{mode}w{window}po_cl.evt",
            "pat.fits.gz",
            "xhdtc.hk",
        ]
        
        for file_for_barycentric_correction in files_for_barycentric_correction:
            output_file_name = file_for_barycentric_correction.replace(
                ".", "_barycentric_correction.", 1
            )
            _=barycorr(infile=f'{xrtpipeline_output_folder}sw{obs_id}{file_for_barycentric_correction}',
                     outfile=f'{xrtpipeline_output_folder}sw{obs_id}{output_file_name}',
                     orbitfiles=f'{xrtpipeline_output_folder}sw{obs_id}sao.fits.gz',
                     clobber=f'clobber={bool_cmd_str(clobber)}',allow_failure=True)



def run_xrtproducts(
    
    raw_data_folder: str,
    data_folder: str,
    output_folder: str,
    obs_id: str = "",
    dry_run=False,
    clobber=True,
    pi_low: float = 30.0,
    pi_high: float = 100.0,
    background_region_file: str = "",
    mode: str = None,
    window: int = None,pix_radius=20,ra=None,dec=None
):
    """Run the commands to extract xrt_products from the data

    Args:
        mode (str): XRT mode
        window (int): which window to use
        data_folder (str): Folder in which the data is contained
        output_folder (str): The folder in which the output from the
        xrtproducts will be stored
        obs_id (str, optional): Observation ID Defaults to "".
        dry_run (bool, optional): If `dry_run==True` then the commands are
        only printed. If `dry_run==False` then the commands are executed.
        Defaults to True.
        clobber (bool, optional): Can files be overwritten if necessary.
        Defaults to True.
        pi_low (float, optional): _description_. Defaults to 30.0.
        pi_high (float, optional): _description_. Defaults to 100.0.

    """
    if not mode or not window:
        modes_list = detect_modes(raw_data_folder+'/xrt/event/')
    else:
        modes_list = [(mode, window)]

    obs_details_dict = observation_details_from_pat(folder=raw_data_folder)
    obs_id = obs_details_dict["OBS_ID"]
    
    if not ra:
        ra = obs_details_dict["object_ra"]
    if not dec:
        dec = obs_details_dict["object_dec"]

    if len(obs_id) == 0:
        obs_id = obs_details_dict["OBS_ID"]

    if output_folder[-1] != "/":
        output_folder += "/"

    if data_folder[-1] != "/":
        data_folder += "/"
    command_list = []

    if isinstance(window, int):
        assert 1 <= window <= 5
        window_str = "w" + str(window)

    file_base = f"{data_folder}sw{obs_id}"

    for mode, window in modes_list:
        window_str = "w" + str(window)

        source_region_file = f"{file_base}x{mode}{window_str}po.reg"
        if not background_region_file:
            # Generate file
            # store filename
            background_region_file = f"{file_base}x{mode}{window_str}po_automatic_background.reg"
            automatic_region_generation(f'{data_folder}')
            
        logfile_name = f"{output_folder}xrtproducts_log_{obs_id}{mode}{window}"
        
        A=xrtproducts(infile=f'{file_base}x{mode}{window_str}po_cl_barycentric_correction.evt',
                      regionfile=f'{source_region_file}',bkgextract='yes',
                      bkgregionfile=f'{background_region_file}',
                      outdir=f'{output_folder}',stemout='DEFAULT',
                      correctlc='yes',expofile=f'{file_base}x{mode}{window_str}po_ex.img',
                      attfile=f'{file_base}pat_barycentric_correction.fits.gz',
                      hdfile=f'{file_base}xhdtc_barycentric_correction.hk',
                      pilow=f'{int(pi_low)}', pihigh=f'{int(pi_high)}',
                      clobber=f'{bool_cmd_str(clobber)}',pcnframe='0',
                      lcfile=f'sw{obs_id}x{mode}{window_str}_lcfile.lc',
                      phafile=f'sw{obs_id}x{mode}{window_str}_phafile.pha' ,
                      bkglcfile=f'sw{obs_id}x{mode}{window_str}_bkglcfile.lc' ,
                      bkgphafile=f'sw{obs_id}x{mode}{window_str}_bkgphafile.pha',
                      allow_failure = True,noprompt=True,
                      )
        with open(logfile_name,'w') as file:
            file.write(str(A))

    # lc_math_command = f"lcmath infile=sw{obs_id}xwtw2posr_corr.lc \
    # bgfile=f"sw{obs_id}xwtw2pobkg.lc outfile=sw{obs_id}xwtw2po_csub.lc
    # multi=1 multb=0.1 addsubr=no"
    # command_list.append(lc_math_command)

    # lcurve_command = f"lcurve nser=1 cfile1=sw{obs_id}xwtw2po_csub.lc
    # window=- dtnb=300 \
    # nbint=f"INDEF outfile=sw{obs_id}xwtw2po_csub.flc plot=no"
    # command_list.append(lcurve_command)

    try:
        with open(logfile_name) as log_file:
            for line in log_file:
                # extract line that contains rmffile location
                if "Name of the input RMF file" in line:
                    # extract string of rmf location
                    rmf_location = (
                        ((line.split(":")[-1]).replace("'", "")).replace(" ", "").replace("\n", "")
                    )

        # copy rmf to destination of xrtproducts ouput
        copy_rmf_string = f"cp '{rmf_location}' \
'{output_folder}{rmf_location.split('/')[-1]}'"
        command_list.append(copy_rmf_string)
    except:
        if os.path.exists(logfile_name):
            raise  Exception(
                f"Log file ({logfile_name}) could not be opened."
            )
        else:
            raise  Exception(
                f"Log file ({logfile_name}) does not exist."
            )


def swiftxrtpipeline(
    raw_data_folder: str,output_folder: str,
    window=None,
    mode=None,
    dry_run: bool = False,
    obs_id: str = "",
    ra=None,
    dec=None,
    radius=None
):
    command_list=[]
        
    if output_folder[-1] != "/":
        output_folder += "/"
        
    xrtpipeline_output_folder = output_folder #+ "apo/"
    if not os.path.isdir(xrtpipeline_output_folder):
        command_list.append(f"mkdir {xrtpipeline_output_folder}")

    if not obs_id:
        obs_id = obsid_from_folder(raw_data_folder)
     
    
    run_xrtpipeline(
        raw_data_folder=raw_data_folder,
        output_folder=xrtpipeline_output_folder,
        dry_run=False,
    )
    print('xrtpipeline DONE!')
    
    run_barycentric_correction(
        data_folder=raw_data_folder,
        xrtpipeline_output_folder=xrtpipeline_output_folder,mode=mode,
        window=window,
    )
    print('barycentric corrections DONE!')
    
    run_xrtproducts(
        mode=mode,
        window=window,
        raw_data_folder=raw_data_folder,
        data_folder=xrtpipeline_output_folder,
        output_folder=output_folder,
        dry_run=True,
        obs_id=obs_id,
    )
    
    print('xrtproducts DONE!')



# TODO xselect output: check for pileup:
# - WT > 100 counts/s
# - PC > 0.6 counts/s


def make_gammapy_compliant(
    folder: str,
    name: str = "",
    pha_file: str = "",
    rmf_file: str = "",
    background_file: str = "",
    arf_file: str = "",
    mode: str = "",
    window: int = -1,
):
    """
    Modify the OGIP files to conform to the standards set out in gammapy.
    This includes:
        - having and EBOUND column (from the rmf file) in the source and
        background .pha files.
        - Including BACKSCAL, QUALITY and GROUPING columns if not already
        present
        - following the naming convention set ou in the gammapy
        OGIPDatasetReader documentation
        - specifying the location of the other file within the source .pha file
        header

    Parameters
    ----------
    folder : str
        Folder in which the files are stored.
    name : str, optional
        Observation name. The default is '', which will result in automatic
        generation.
    pha_file : str, optional
        Name of the pha file. If empty, name will be infered.
        The default is ''.
    rmf_file : str, optional
        Name of the rmf file. If empty, name will be infered.
        The default is ''.
    background_file : str, optional
        Name of the background file. If empty, name will be infered.
        The default is ''.
    arf_file : str, optional
        Name of the arf file. If empty, name will be infered.
        The default is ''.

    Returns
    -------
    None.

    """

    gammapy_compliant_names = [
        "pha_obs{name}.fits",
        "bkg_obs{name}.fits",
        "arf_obs{name}.fits",
        "rmf_obs{name}.fits",
    ]

    if not rmf_file:
        try:
            rmf_file = fnmatch.filter(os.listdir(folder), f"*{mode}*.rmf")[0]
        except:
            raise Exception(
                f"RMF file could not be found in automatically in {folder}.\
                    \nPlease specify which file to use with the `rmf_file` \
                        argument."
            )
    if folder[-1] != "/":
        folder += "/"

    with fits.open(folder + rmf_file) as hdul:
        EBOUNDS = hdul["EBOUNDS"].copy()

    if not pha_file:
        pha_file = "phafile.pha"

    if not background_file:
        background_file = "bkgphafile.pha"

    if not arf_file:
        try:
            arf_file = fnmatch.filter(os.listdir(folder), f"*{mode}*{str(window)}*.arf")[0]
        except:
            raise Exception(
                f"ARF file *{mode}*{str(window)}*.arf could not be found in automatically in {folder}\
                .\nPlease specify which file to use with the `arf_file` \
                    argument."
            )

    name = "_swift_obs"

    BACKSCAL = -1

    for file, new_filename in zip(
        [pha_file, background_file, arf_file, rmf_file],
        gammapy_compliant_names,
    ):
        with fits.open(folder + file) as file_hdu:

            if file == pha_file:
                if BACKSCAL < 0:
                    BACKSCAL = file_hdu["SPECTRUM"].header["BACKSCAL"]
                    file_hdu["SPECTRUM"].header["BACKFILE"] = f"bkg_obs{name}.fits"
                    file_hdu["SPECTRUM"].header["RESPFILE"] = f"rmf_obs{name}.fits"
                    file_hdu["SPECTRUM"].header["ANCRFILE"] = f"arf_obs{name}.fits"
            if file == pha_file or file == background_file:
                if not name:
                    name = (
                        "_"
                        + file_hdu["Primary"].header["TELESCOP"]
                        + file_hdu["Primary"].header["INSTRUME"]
                        + "_"
                        + file_hdu["Primary"].header["OBS_ID"]
                    )
                file_hdu.append(EBOUNDS)

                # TODO store values in the BACKFILE, ANCRFILE and RESPFILE to
                # point to respective files

                additonal_columns = []

                if "QUALITY" not in file_hdu["SPECTRUM"].columns.names:
                    additonal_columns.append(
                        fits.Column(
                            name="QUALITY",
                            format="L",
                            array=np.zeros(len(file_hdu["SPECTRUM"].data["CHANNEL"])),
                        )
                    )

                if "GROUPING" not in file_hdu["SPECTRUM"].columns.names:
                    additonal_columns.append(
                        fits.Column(
                            name="GROUPING",
                            format="I",
                            array=np.ones(len(file_hdu["SPECTRUM"].data["CHANNEL"])),
                        )
                    )

                if "BACKSCAL" not in file_hdu["SPECTRUM"].columns.names:
                    additonal_columns.append(
                        fits.Column(
                            name="BACKSCAL",
                            format="D",
                            array=BACKSCAL * np.ones(len(file_hdu["SPECTRUM"].data["CHANNEL"])),
                        )
                    )

                if len(additonal_columns) > 0:
                    file_hdu = append_column(file_hdu["SPECTRUM"], additonal_columns)

            new_filename = new_filename.format(name=name)
            # Write changes to new_file
            file_hdu.writeto(
                folder + "gammapy_compliant/" + new_filename,
                overwrite=True,
                output_verify="fix",
                checksum=True,
            )


def append_column(original_table, columns):

    original_cols = original_table.columns

    if isinstance(columns, list):
        new_cols = fits.ColDefs(columns)
    else:
        new_cols = fits.ColDefs([columns])
    new_hdu = fits.BinTableHDU.from_columns(original_cols + new_cols)
    return new_hdu

def obs_id_from_fits(hdu,obs_id):
    if not obs_id:
        try:
            return hdu[0].header['OBS_ID']
        except:
            return None
    return None


def generate_diagnostic_plot(data_folder:str,pha_folder:str,mode:str=None,window:str=None,):
    #TODO Make automatic mode and window detection possible
    
    
    fig, ax = plt.subplots(2, 3,figsize=(11.69,8.27),width_ratios=[1, 1,1])
    
    if data_folder[-1] != '/':data_folder.append('/')
    if pha_folder[-1] != '/':pha_folder+=('/')
    
    event_folder=data_folder+'/xrt/event/'
    house_keeping_folder=data_folder+'/xrt/hk/'
    
    if house_keeping_folder[-1] != '/':house_keeping_folder.append('/')
    
    obs_id=None
    
    #RAWX vs RAWY
    if event_folder[-1] != '/':event_folder+='/'
    current_ax=ax[1][0]
    current_ax.set_title('DETECTOR SPACE')
    current_ax.set_xlabel('RAWX')
    current_ax.set_ylabel('RAWY')
    marker=','
    
    raw_events_file=fnmatch.filter(os.listdir(event_folder), f"*{mode}*{str(window)}*po_uf.evt*")[0]
    clean_events_file=fnmatch.filter(os.listdir(event_folder), f"*{mode}*{str(window)}*po_cl.evt*")[0]
    try:
        with fits.open(event_folder+raw_events_file) as hdu:
            current_ax.scatter(hdu['EVENTS'].data['RAWX'],hdu['EVENTS'].data['RAWY'],s=1,c='orange',alpha=0.1,label='All',marker=marker)
            obs_id=obs_id_from_fits(hdu,obs_id)
            
        with fits.open(event_folder+clean_events_file) as hdu:
            current_ax.scatter(hdu['EVENTS'].data['RAWX'],hdu['EVENTS'].data['RAWY'],s=1,c='b',alpha=1,label='Cleaned',marker=marker)
            current_ax.scatter(hdu['BADPIX'].data['RAWX'],hdu['BADPIX'].data['RAWY'],s=1,c='hotpink',alpha=1,label='Bad Pixels',marker=marker)
            obs_id=obs_id_from_fits(hdu,obs_id)
            
    except:
        current_ax.text(.5,.5,'ERROR')
    
    
    current_ax.legend(loc='upper right',fontsize=fontsize_legend)
    
    #X-Y and source and bkg regions

    marker="."
      
    with fits.open(event_folder+clean_events_file) as hdu:
        obs_id=obs_id_from_fits(hdu,obs_id)
        w = wcs.WCS(naxis=2)
        w.wcs.crpix = [hdu['EVENTS'].header['TCRPX2'], hdu['EVENTS'].header['TCRPX3']]
        w.wcs.cdelt = [hdu['EVENTS'].header['TCDLT2'], hdu['EVENTS'].header['TCDLT3']]
        w.wcs.crval = [hdu['EVENTS'].header['TCRVL2'], hdu['EVENTS'].header['TCRVL3']]
        w.wcs.ctype = [hdu['EVENTS'].header['TCTYP2'], hdu['EVENTS'].header['TCTYP3']]
        w.wcs.radesys = hdu['EVENTS'].header['RADECSYS']
        w.wcs.equinox = hdu['EVENTS'].header['EQUINOX']
        
        x_range=np.max(hdu['EVENTS'].data['X'])-np.min(hdu['EVENTS'].data['X'])
        y_range=np.max(hdu['EVENTS'].data['Y'])-np.min((hdu['EVENTS'].data['Y']))
        
        ax[0][0].remove()
        ax[0][0] = fig.add_subplot(2, 3, 1, projection=w)
        current_ax=ax[0][0]
        
        
        
        n_counts=len(hdu['EVENTS'].data['X'])
         
        current_ax.hist2d(hdu['EVENTS'].data['X'],hdu['EVENTS'].data['Y'],
                          bins=[x_range,y_range],density=True,
                          norm = colors.LogNorm(),cmap='magma',label='Counts')
        plt.scatter([hdu[0].header['RA_OBJ']]*u.deg,[hdu[0].header['DEC_OBJ']]*u.deg,label='OBJECT',transform=current_ax.get_transform('world'),marker='+',c='lime',zorder=0)

    current_ax.set_title('SKY SPACE')
    current_ax.set_xlabel('RA')
    current_ax.set_ylabel('DEC')
    current_ax.coords['ra'].set_format_unit('deg', decimal=True, show_decimal_unit=True)
    current_ax.coords['dec'].set_format_unit('deg', decimal=True, show_decimal_unit=True)
    current_ax.grid(color='gray', ls='dashed')
    
    #Regions
    
    region_files=fnmatch.filter((os.listdir(pha_folder)), f'*{mode}*{window}*.reg*')
    
    for region_file in region_files:
        try:
            region = Regions.read(pha_folder+region_file)
            c=src_col
            hatch='/////'
            
            if 'background' in region_file.lower() or 'bkg' in region_file.lower():
                c=bkg_col
                hatch='\\\\\\\\'
                zorder=5
            region.regions[0].plot(ax=current_ax,edgecolor=c,alpha=0.8,)#, label=region_file)
        except:
            continue

    current_ax.legend(loc='upper right',fontsize=fontsize_legend)
    
    #RA-DEC for pointing times
    #With FOV and rotation
    ax[0][2].remove()
    ax[0][2] = fig.add_subplot(2, 3, 3,projection=w)
    current_ax=ax[0][2]
    
    
    plt.scatter([0,0,1000,1000],[0,1000,0,1000],s=0,transform=current_ax.get_transform('pixel'),marker='1')
    #radec=w.pixel_to_world([[0,0,600,600],[0,600,600,0]],[0])
    
        
    with fits.open(house_keeping_folder+fnmatch.filter(os.listdir(house_keeping_folder),'*hd.hk*')[0]) as hdu:
        
        ra=hdu['FRAME'].data['RA'][:-1]
        dec=hdu['FRAME'].data['DEC'][:-1]
        roll=hdu['FRAME'].data['Roll'][:-1]
        settled=(hdu['FRAME'].data['Settled'][:-1].T)[0]
        
        #TODO Possibly change to restrict operating mode and window plot
        
        current_ax.plot(ra[settled]*u.deg,dec[settled]*u.deg,transform=current_ax.get_transform('world'),label='Pointing Centre')
        
        label='FOV'
        for pointing in zip(ra[settled],dec[settled],roll[settled]):
            
            center=SkyCoord(pointing[0]*u.deg, pointing[1]*u.deg,frame='fk5')
            region_sky = RectangleSkyRegion(center=center,
                                      width=23.6*u.arcmin, height=23.6*u.arcmin,
                                      angle=(pointing[2]-90.0) * u.deg)
            pixel_region = region_sky.to_pixel(w)
            pixel_region.plot(ax=current_ax,label=label)
            label=None#Only incluides label on the first drawing
            
        current_ax.scatter([hdu[0].header['RA_OBJ']]*u.deg,[hdu[0].header['DEC_OBJ']]*u.deg,label='OBJECT',transform=current_ax.get_transform('world'),marker='+',c='lime')
        
        
        object_coords=[[hdu[0].header['RA_OBJ']]*u.deg,[hdu[0].header['DEC_OBJ']]*u.deg]
        obs_times=[hdu[0].header['DATE-OBS'],hdu[0].header['DATE-END']]

    
    current_ax.coords['ra'].set_format_unit('deg', decimal=True, show_decimal_unit=True)
    current_ax.coords['dec'].set_format_unit('deg', decimal=True, show_decimal_unit=True)
    current_ax.grid(color='gray', ls='dashed')

    
    current_ax.set_xlabel('RA')
    current_ax.set_ylabel('DEC')

    current_ax.set_title('TELESCOPE POINTING')
    
    current_ax.legend(loc='upper right',fontsize=fontsize_legend)
    
    
    
    
    
    
    #COUNTS vs CHANNEL FOR SOURCE AND BACKGROUND

    current_ax=ax[0][1]
    current_ax.set_title('COUNTS')
    current_ax.set_xlabel('CHANNEL')
    current_ax.set_ylabel('COUNTS')
    
    for file in fnmatch.filter(os.listdir(pha_folder),f'*{mode}*{window}*phafile.pha*'):
        with fits.open(pha_folder+file) as hdu:
            c=None
            if 'background' in file.lower() or 'bkg' in file.lower():
                c=bkg_col
            elif 'phafile.pha' in file:
                c=src_col  
            current_ax.step(hdu['SPECTRUM'].data['CHANNEL'],hdu['SPECTRUM'].data['COUNTS'],label=file,where='mid',color=c)
            
    current_ax.legend(loc='upper right',fontsize=fontsize_legend)
    
    #COUNT RATE vs TIME
    current_ax=ax[1][1]
    current_ax.set_title('COUNT RATE')
    current_ax.set_xlabel('TIME')
    current_ax.set_ylabel('RATE')
    
    for file in fnmatch.filter(os.listdir(pha_folder),f'*{mode}*{window}*lcfile.lc*'):
        with fits.open(pha_folder+file) as hdu:
            c=src_col
            zorder=5
            if 'background' in file.lower() or 'bkg' in file.lower():
                c=bkg_col
                zorder=52
            current_ax.errorbar(x=hdu['RATE'].data['TIME'],y=hdu['RATE'].data['RATE'],
                                yerr=hdu['RATE'].data['ERROR'],label=file,c=c,zorder=zorder)
    current_ax.legend(loc='upper right',fontsize=fontsize_legend)
       
    current_ax=ax[1][2]
    current_ax.text(x=0,y=1,s=f'Observation ID: {obs_id}\nMode:{mode}\
                    \nWindow:{window}\nRA  :  {object_coords[0][0]}\
                    \nDEC:  {object_coords[1][0]}\nTotal counts:{n_counts}\n\
                    \nSTART:\n     {obs_times[0]}\
                    \nEND:  \n     {obs_times[1]}\
                    ',verticalalignment='top')
    current_ax.set_xlim(0,1)
    current_ax.set_ylim(0,1)
    current_ax.axis('off')
    fig.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.savefig(pha_folder+f'sw{obs_id}x{mode}w{window} Summary.png')
    plt.show()

if __name__ == "__main__":
    BACKSCAL = 1.25e-1
    #generate_diagnostic_plot(event_folder='/home/user/Documents/leptohadronic-fitting/swiftpipeline/data/00011184023/xrt/event/',
                            #mode='wt',window='2',pha_folder='/home/user/Documents/leptohadronic-fitting/swiftpipeline/data/xrtpipeline_output/00011184023/xproducts_output')
    #generate_diagnostic_plot(data_folder='/home/user/Documents/Gammapy_tests/swiftxrt_data/00030795038/',
                             #mode='wt',window='2',pha_folder='/home/user/Documents/Gammapy_tests/swiftxrt_data/script/apo/')
    # make_gammapy_compliant("/home/user/Documents/Gammapy_tests/swiftxrt_data/00030795038_processed")
    
    data_folder='/home/user/Documents/leptohadronic-fitting/swiftpipeline/data/00011184023/'
    output_folder='/home/user/Documents/leptohadronic-fitting/swiftpipeline/data/00011184023_apo/'
    
    #swiftxrtpipeline(raw_data_folder=data_folder,output_folder=output_folder)
    
    generate_diagnostic_plot(data_folder=data_folder,mode='wt',window='2',pha_folder=output_folder)