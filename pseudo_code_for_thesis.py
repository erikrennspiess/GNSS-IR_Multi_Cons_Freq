def orbit_downloader(date, orbit_type, orbit_base_url, orbit_storage_directory):
    # this function retrieves a satellites' orbit information (SP3)
    gps_date_for_url = gps_week_day_calculator(start_date, end_date)
    for i in orbit_base_url:
        sp3_url = format_base_url + gps_date_for_url
        download_file = wget.download(sp3_url)

def orbit_info_extractor(sp3_file_location, satellites, station_lat, station_lon, station_h):
    # this function extracts necessary X,Y,Z coordinates from navigation files based on date and station location
    df = nav []
    for i in sp3_file_location:
        navigation_data = georinex.load(navigation_file)
        navigation_data = georinex.nav.select(satellites)

        #define coordinate and location variables:
        x_km = navigation_data['x'] * 1000 #convert from meters to km
        y_km = navigation_data['y'] * 1000 #convert from meters to km
        z_km = navigation_data['z'] * 1000 #convert from meters to km

        lat = station_lat
        lon = station_lon
        h = station_h

        #convert from ECEF to Azimuth, Elevation & Range:
        az, el, srange = pm.ecef2aer(x_km,y_km,z_km, lat, lon, h, ell=pm.Ellipsoid('wgs84'), deg=True)

    return df

def interpolation_function(time, azimuth, elevation):
    # for the interpolation function we need to start the algorithm from index 6 and stop the algorithm at -7 from
    # the last position.
    for i in np.arange(start=6, stop=length_of_data_set - 7, step = 1):
        azimuth_interpolation = scipy.interpolate.lagrange(azimuth_reference_values)
        elevation_interpolation = scipy.interpolate.lagrange(elevation_reference_values)

    return azimuth_interpolated_values, elevation_interpolated_values

def main():
   # please input information for: orbit type, frequencies, gps satellite keys, galileo satellite keys
   # station lat, lon and ellipsoidal height, minimum reflector height, maximum reflector height, azimuth parameters,

    for i in date_range(start_date, end_date):
        collect_orbit_info = downloaded_sp3_info
        collect_observation_info = observation_directory

        if observations_are_valid
            continue

        if navigation_is_valid
           continue

        for j in observation_list:
            height_model_values = 2 * np.sin(EL * np.pi / 180.) / signal_wavelength[obs_type_list.index(obs_type)]
            detrended_snr = snr - np.polyval(p, x)
            pdg = signal.lombscargle(x, detrended_snr, 2 * np.pi * x_freq)
            pdg = pdg * 100 / len(detrended_snr)
            peaks, _ = signal.find_peaks(pdg)

        results = [[d, satellite, obs_type, np.nanmean(EL),
                    np.nanmean(AZ), oscillation_power, reflector_height]]
        results = pd.DataFrame(results,
                               columns=["time", "sv", "obs_type", "mean_elevation", "mean_azimuth",
                                        "oscillation_power",
                                        "reflector_height"])

