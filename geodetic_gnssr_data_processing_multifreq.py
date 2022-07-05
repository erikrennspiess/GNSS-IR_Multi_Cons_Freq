# The program's inputs: a time period (start and end date), the directory path of geodetic receiver files, and
# the URL for retrieving satellites' orbit information (SP3)
# The program's output: a Pandas dataframe with 3 columns respectively for the datetime, PRN number, estimated height
import time
import re
import pandas as pd
import numpy as np
import wget
import unlzw3
from pathlib import Path
import os
import georinex as gr
import glob
import pymap3d as pm
import scipy.interpolate
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy import stats
import gzip
import ftplib


def gps_week_day_calculator(target_date, days_shift=0):
    days_since_gps_ref_epoch = pd.to_datetime(target_date) - pd.to_datetime("1980-01-06 00:00:00")
    gps_week = int(np.floor((days_since_gps_ref_epoch.days + days_shift) / 7))
    gps_day_of_week = int(days_since_gps_ref_epoch.days + days_shift - gps_week * 7)
    return gps_week, gps_day_of_week


# # testing the function gps_week_day_calculator:
# gps_week, gps_day_of_week = gps_week_day_calculator("2018-05-28")
# gps_week, gps_day_of_week = gps_week_day_calculator("2018-05-28 18:27:58")
# gps_week, gps_day_of_week = gps_week_day_calculator("2018-05-28", +1)  # one day after the specified date
# gps_week, gps_day_of_week = gps_week_day_calculator("2018-05-28", -2)  # two days before the specified date


def orbit_downloader(orbit_date, orbit_type, orbit_download_base_urls, orbit_storage_dir):
    # first, we convert the orbit_date to gps week and day of the week
    gps_week, gps_day_of_week = gps_week_day_calculator(orbit_date)
    orbit_date = pd.to_datetime(orbit_date)
    day_of_year = str(orbit_date.dayofyear).zfill(3)

    uncompressed_orbit_file_path = []
    # then, we construct the download url based on the week number and day of the week
    for base_url in orbit_download_base_urls:
        download_url = base_url.format(AAA=orbit_type, WWWW=gps_week, DDD=day_of_year, D=gps_day_of_week,
                                       YYYY=orbit_date.year)

        existing = glob.glob(os.path.join(orbit_storage_dir, "*" + str(gps_week) + str(gps_day_of_week) + ".sp3"))
        if len(existing) > 0:
            return existing[0]
        try:
            # downloading the sp3 file
            print("downloading " + download_url)

            download_filename = os.path.split(download_url)

            if re.match("ftp://", download_url):
                # urllib.urlretrieve(download_url, download_filename[-1])

                ftp = ftplib.FTP()
                download_filename_split = download_filename[0].split('/')
                ftp.connect(download_filename_split[2])
                ftp.login()
                ftp.cwd("/".join(download_filename_split[3:]))
                ftp.retrbinary(f"RETR {download_filename[-1]}", download_filename[-1])
            else:
                wget.download(download_url)

            download_filename_split = download_filename[-1].split('.')
            if download_filename_split[-1] == 'Z':
                uncompressed_data = unlzw3.unlzw(Path(download_filename[-1]))

            elif download_filename_split[-1] == 'gz':
                gz_file = gzip.open(download_filename[-1])
                uncompressed_data = gz_file.read()
                gz_file.close()

            # decompressing the "sp3.Z" file
            uncompressed_orbit_file_path = orbit_storage_dir + orbit_type + str(gps_week) + str(
                gps_day_of_week) + ".sp3"
            file = open(uncompressed_orbit_file_path, "wb")
            file.write(uncompressed_data)
            file.close()

            # removing the "sp3.Z" file
            os.remove(download_filename[-1])

            # returning the full path of the downloaded and decompressed sp3 file
            return uncompressed_orbit_file_path

        except:
            continue

    return uncompressed_orbit_file_path


# # testing the function orbit_downloader:
# orbit_download_base_url = "https://igs.bkg.bund.de/root_ftp/IGS/products/orbits/"
# orbit_storage_dir = "M:\\Research\\Erik\\orbit\\"
# orbit_type = "igs"  # rapid: igr, ultra_rapid: igu
# orbit_date = "2018-05-28"
# downloaded_sp3_path = orbit_downloader(orbit_date, orbit_type, orbit_download_base_url, orbit_storage_dir)


def orbit_info_extractor(sp3_files_list, satellite_keys, station_lat, station_lon, station_h):
    df_nav = []
    for rinex_nav_file in sp3_files_list:
        nav = gr.load(rinex_nav_file)
        nav = nav.sel(sv=np.in1d(nav["sv"], satellite_keys))
        # nav = nav.drop(["clock", "velocity", "dclock"])
        df = nav.to_dataframe()
        df.reset_index(inplace=True)
        df_pivot = df.pivot(index=["sv", "time"], columns="ECEF",
                            values="position")  # pivots table to time, x, y, z columns
        df_nav.append(df_pivot)

    df_nav = pd.concat(df_nav)

    # sp3 files uses km and pymap3d uses meters for input, so need to convert:
    # x values:
    x_km = df_nav['x']
    x = np.array(x_km)
    x_m = x * 1000

    # y values:
    y_km = df_nav['y']
    y = np.array(y_km)
    y_m = y * 1000

    # z values:
    z_km = df_nav['z']
    z = np.array(z_km)
    z_m = z * 1000

    # station coordinates for onsala converted from minutes to degrees, change for other receiver location(s)
    lat = station_lat
    long = station_lon
    h = station_h

    # convert from Earth Center Earth Fixed to Azimuth, Elevation, Range
    az, el, srange = pm.ecef2aer(x_m, y_m, z_m, lat, long, h, ell=pm.Ellipsoid('wgs84'), deg=True)

    # attach results to dataframe
    df_nav['azimuth'] = np.array(az)
    df_nav['elevation'] = np.array(el)
    df_nav = df_nav.sort_index(0)
    return df_nav


def interpolation_function(time_array, az_array, el_array, time_query):
    az_query = np.empty(np.shape(time_query))
    el_query = np.empty(np.shape(time_query))
    for i in np.arange(start=6, stop=len(time_array) - 7, step=1):
        x_ref = time_array[i - 6:i + 6]
        ind_query = np.where((time_query >= x_ref[5]) & (time_query >= x_ref[6]))
        if np.size(ind_query) == 0:
            continue

        x_ref_mean = np.mean(time_array[i - 6:i + 6])
        t = time_query[ind_query] - x_ref_mean
        x_ref = x_ref - x_ref_mean
        y_ref = az_array[i - 6:i + 6]
        z_ref = el_array[i - 6:i + 6]
        p = scipy.interpolate.lagrange(x_ref / 60,
                                       y_ref)  # division by 60 to convert minutes to make x_values even smaller
        p_z = scipy.interpolate.lagrange(x_ref / 60, z_ref)
        x_query = t
        y_query = p(x_query / 60)  # division by 60 to convert minutes
        z_query = p_z(x_query / 60)
        az_query[ind_query] = y_query
        el_query[ind_query] = z_query

    return az_query, el_query


def read_rinex_obs(rinex_obs_files, obs_types, satellite_keys):
    df_obs = []
    for obs_file in rinex_obs_files:
        try:
            print(f"reading observations from: {obs_file}")
            obs = gr.load(obs_file)
            obs = obs.sel(sv=np.in1d(obs["sv"], satellite_keys))
            obs = obs[obs_types]
            df = obs.to_dataframe()
            df.reset_index(inplace=True)
            df = df.dropna()
            df_obs.append(df)

        except:
            print("failt to read observations.")

    if len(df_obs) > 0:
        df_obs = pd.concat(df_obs)
        df_obs = df_obs.sort_values(["sv", "time"])

    return df_obs


def main():
    start_date = input("Enter a start date, e.g. 2021-01-01: ")
    end_date = input("Enter an end date, e.g. 2021-01-31: ")
    orbit_type = "igs"
    obs_type_list = ["S1C", "S2X", "S5X"]
    satellite_keys_gps = ["G" + str(i).zfill(2) for i in range(1, 33)]
    satellite_keys_galileo = ["E" + str(i).zfill(2) for i in range(1, 37)]
    satellite_keys = satellite_keys_gps + satellite_keys_galileo

    # Onsala GNSS-R station coordinate
    lat = 57.393016
    lon = 11.91369
    h = 40

    # Parameters for Lombscargle
    min_reflector_height = 1
    max_reflector_height = 6
    x_freq = np.arange(start=min_reflector_height, stop=max_reflector_height, step=0.01)
    signal_wavelength = [0.1905, 0.2445, 0.2548]  # in meters, must correspond to obs_types

    # selecting proper azimuth and elevation angles associated with reflections from sea surface
    azimuth_start = 60
    azimuth_end = 220
    elevation_limit = 60

    # moving step and time window for selecting observations
    moving_step = 300  # seconds
    moving_window = 1800  # seconds

    obs_dir = "\\\\ibmrs01.ibm.ntnu.no\\SatelliteData\\Onsala\\rinex\\"
    orbit_download_base_url = ["http://ftp.aiub.unibe.ch/CODE_MGEX/CODE/{YYYY}/COM{WWWW}{D}.EPH.Z",
                               "ftp://igs.ensg.ign.fr/pub/igs/products/mgex/{WWWW}/WUM0MGXFIN_{YYYY}{DDD}0000_01D_05M_ORB.SP3.gz"]

    orbit_storage_dir = "M:\\Research\\Erik\\orbit\\"
    output_csv_path = f"M:\\Research\\Erik\\results_{start_date}_{end_date}.csv"

    for obs_date in pd.date_range(start=start_date, end=end_date):

        # First: collecting all the orbit info needed for one day before and one day after the observation date
        sp3_files_list = []
        for orbit_date in pd.date_range(start=obs_date + pd.Timedelta(-1, unit='d'),
                                        end=obs_date + pd.Timedelta(+1, unit='d')):
            downloaded_sp3_path = orbit_downloader(orbit_date, orbit_type, orbit_download_base_url, orbit_storage_dir)
            sp3_files_list.append(downloaded_sp3_path)

        nav = orbit_info_extractor(sp3_files_list, satellite_keys, lat, lon, h)

        # finding all the observation files on the obs_date
        obs_files_list = glob.glob(os.path.join(obs_dir, "*" + obs_date.strftime('%Y%m%d') + "*"))
        if len(obs_files_list) == 0:
            continue

        obs_files_list = sorted(obs_files_list)
        obs = read_rinex_obs(obs_files_list, obs_type_list, satellite_keys)

        if obs["time"].size < 10:  # check if the valid observations are long enough for processing
            continue

        for obs_type in obs_type_list:
            for satellite in satellite_keys:
                ind_sat = obs["sv"] == satellite
                selected_obs_sat = obs[ind_sat]
                if selected_obs_sat["time"].size < 10:  # check if the valid observations are long enough for processing
                    continue

                start_obs_time = np.min(selected_obs_sat["time"])
                start_obs_time = pd.Series(start_obs_time).dt.round(str(moving_step) + "S")

                end_obs_time = np.max(selected_obs_sat["time"])
                end_obs_time = pd.Series(end_obs_time).dt.round(str(moving_step) + "S")

                for d in pd.date_range(start=start_obs_time[0], end=end_obs_time[0], freq=str(moving_step) + "S"):
                    ind_time = (selected_obs_sat["time"] >= d - pd.Timedelta(str(moving_window / 2) + "s")) & \
                               (selected_obs_sat["time"] <= d + pd.Timedelta(str(moving_window / 2) + "s"))

                    selected_obs_time = selected_obs_sat[ind_time]
                    if selected_obs_time[
                        "time"].size < 10:  # check if the valid observations are long enough for processing
                        continue

                    time_query = ((selected_obs_time["time"] - pd.to_datetime("1980-01-06 00:00:00")) / pd.Timedelta(
                        "1s"))

                    start_obs_time = np.min(obs["time"])
                    end_obs_time = np.max(obs["time"])

                    selected_nav_ind = (nav.index.get_level_values(0) == satellite) & \
                                       (nav.index.get_level_values(1) >= start_obs_time - pd.Timedelta("3h")) & \
                                       (nav.index.get_level_values(1) <= end_obs_time + pd.Timedelta("3h"))

                    selected_nav = nav[selected_nav_ind]

                    nav_time_array = np.array(
                        (selected_nav.index.get_level_values(1) - pd.to_datetime("1980-01-06 00:00:00")) / pd.Timedelta(
                            "1s"))
                    az_array = selected_nav["azimuth"]
                    el_array = selected_nav["elevation"]
                    AZ, EL = interpolation_function(nav_time_array, az_array.values, el_array.values, time_query.values)

                    ind_sea_reflection = (AZ >= azimuth_start) & (AZ <= azimuth_end) & (
                            EL <= elevation_limit)
                    selected_obs_sea = selected_obs_time[ind_sea_reflection]
                    if selected_obs_sea["time"].size < 10 or (
                            np.max(selected_obs_sea["time"]) - np.min(selected_obs_sea["time"])) < pd.Timedelta(
                        str(moving_window * 0.9) + "S"):  # check if the valid observations are long enough for processing
                        continue

                    try:
                        AZ = AZ[ind_sea_reflection]
                        EL = EL[ind_sea_reflection]
                        print(f"retrieving sea level height at: {d}")
                        x = 2 * np.sin(EL * np.pi / 180.) / signal_wavelength[obs_type_list.index(obs_type)]
                        snr = selected_obs_sea[obs_type].values
                        snr = np.reshape(snr, (-1,))
                        p = np.polyfit(x, snr, 2)
                        detrended_snr = snr - np.polyval(p, x)
                        pdg = signal.lombscargle(x, detrended_snr, 2 * np.pi * x_freq)
                        pdg = pdg * 100 / len(detrended_snr)
                        peaks, _ = signal.find_peaks(pdg)
                        pdg_peaks = pdg[peaks]
                        i_sorted_peaks = np.argsort(pdg[peaks])[::-1]
                        peaks = peaks[i_sorted_peaks]
                        pdg_peaks = pdg_peaks[i_sorted_peaks]
                        reflector_height = x_freq[peaks[0]]
                        oscillation_power = pdg_peaks[0]

                        results = [[d, satellite, obs_type, np.nanmean(EL),
                                    np.nanmean(AZ), oscillation_power, reflector_height]]

                        results = pd.DataFrame(results,
                                               columns=["time", "sv", "obs_type", "mean_elevation", "mean_azimuth",
                                                        "oscillation_power",
                                                        "reflector_height"])
                        results.to_csv(output_csv_path, float_format='%.3f', mode='a', index=False,
                                       header=not os.path.exists(output_csv_path))
                        print(f"estimated reflector height is {reflector_height} meter")

                    except:
                        print("reflector height retrieval was not successful!")


if __name__ == "__main__":
    start_processing = time.time()
    main()
    end_processing = time.time()
    print(end_processing - start_processing)