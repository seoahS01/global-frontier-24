"""
Source of this module: https://github.com/building-energy/epw by Steven Firth and Patrick Kastner (license MIT) 
"""

import csv
import pandas as pd

class Weather:
    """
    A wrapper for EnergyPlus weather files (.epw)
    """

    def __init__(self):
        self.headers = {}
        self.dataframe = None


    def read(self, file_path):
        """
        Read the given epw file.
        """
        self.headers = self._read_metadata(file_path)
        self.dataframe = self._read_data(file_path)


    def _read_metadata(self, file_path):
        """
        Read the headers of the given epw file.
        """

        d = {}
        with open(file_path, newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',', quotechar='"')
            for row in csvreader:
                if row[0].isdigit():
                    break
                else:
                    d[row[0]] = row[1:]
        return d


    def _read_data(self,fp):
        """
        Read the climate data of the given epw file.
        """

        names=['Year',
               'Month',
               'Day',
               'Hour',
               'Minute',
               'Data Source and Uncertainty Flags',
               'Dry Bulb Temperature',
               'Dew Point Temperature',
               'Relative Humidity',
               'Atmospheric Station Pressure',
               'Extraterrestrial Horizontal Radiation',
               'Extraterrestrial Direct Normal Radiation',
               'Horizontal Infrared Radiation Intensity',
               'Global Horizontal Radiation',
               'Direct Normal Radiation',
               'Diffuse Horizontal Radiation',
               'Global Horizontal Illuminance',
               'Direct Normal Illuminance',
               'Diffuse Horizontal Illuminance',
               'Zenith Luminance',
               'Wind Direction',
               'Wind Speed',
               'Total Sky Cover',
               'Opaque Sky Cover (used if Horizontal IR Intensity missing)',
               'Visibility',
               'Ceiling Height',
               'Present Weather Observation',
               'Present Weather Codes',
               'Precipitable Water',
               'Aerosol Optical Depth',
               'Snow Depth',
               'Days Since Last Snowfall',
               'Albedo',
               'Liquid Precipitation Depth',
               'Liquid Precipitation Quantity']

        first_row = self._first_row_with_climate_data(fp)
        df = pd.read_csv(fp,
                         skiprows=first_row,
                         header=None,
                         names=names)

        return df


    def _first_row_with_climate_data(self,fp):
        """Finds the first row with the climate data of an epw file
        """

        with open(fp, newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',', quotechar='"')
            for i,row in enumerate(csvreader):
                if row[0].isdigit():
                    break
        return i


    def write(self, file_path):
        """
        Write data in the given epw file.
        """

        with open(file_path, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile,
                                   delimiter=',',
                                   quotechar='"',
                                   quoting=csv.QUOTE_MINIMAL)

            for k, v in self.headers.items():
                csvwriter.writerow([k] + v)

            for row in self.dataframe.itertuples(index=False):
                csvwriter.writerow(i for i in row)