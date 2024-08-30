from os.path import join
import json

import numpy as np


class ParamPhysScaler:
    """
    Class that wraps the methods to fit, apply and inverse apply the 
    scalers for the physical parameters T_eff, mass, lum, rho and radius.
    Each variable is hardcoded, as the normalization was achieved by visually
    inspecting the distributions.
    """
    def __init__(self,
                 param,
                 mask_value=0):
        self.mask = None
        self.param = param

        self.mask_value = mask_value

        self.mean = None
        self.std = None
        self.min = None

    def __transform_teff(self, X):
        # x_trans = (x-3000.0)/1000.0
        self.mean = np.mean(X)
        self.std = np.std(X)
        X_trans = (X - self.mean) / self.std

        return X_trans

    def __inverse_transform_teff(self, x_trans):
        x_trans = np.asarray(x_trans)
        x = np.multiply(x_trans, self.std) + self.mean
        return x

    def __load_teff(self, path_scaler: str)->None:
        """
        Load the scalers for effective temperature T_eff.
        
        Inputs:
        path_scaler: Path to the json file
        """
        # Open the file and assign the data to scaler_data object
        with open(path_scaler, 'r') as file:
            scaler_data = json.load(file)

        # Assign the stored elements to the corresponding properties
        self.mean = scaler_data['mean']
        self.std = scaler_data['std']

    def __transform_lum(self, x):
        x_trans = np.log10(x + 0.01) + 4.0
        return x_trans

    def __inverse_transform_lum(self, x_trans):
        x_trans = np.asarray(x_trans)
        x = np.power(10.0, x_trans - 4.0) - 0.01
        return x

    def __load_lum(self, path_scaler: str)->None:
        """
        Load the scalers for lum.
        
        Inputs:
        path_scaler: Path to the json file
        """
        # Open the file and assign the data to scaler_data object
        #with open(path_scaler, 'r') as file:
        #    scaler_data = json.load(file)

        # No files are needed

    def __transform_mass(self, X):
        x_trans = X
        return x_trans

    def __inverse_transform_mass(self, x_trans):
        x_trans = np.asarray(x_trans)
        x = x_trans
        return x

    def __load_mass(self, path_scaler: str)->None:
        """
        Load the scalers for mass.
        
        Inputs:
        path_scaler: Path to the json file
        """
        # Open the file and assign the data to scaler_data object
        #with open(path_scaler, 'r') as file:
        #    scaler_data = json.load(file)

        # No files are needed

    def __transform_rho(self, X):
        x_trans = np.log10(X + 0.01) + 3.0
        return x_trans

    def __inverse_transform_rho(self, x_trans):
        x_trans = np.asarray(x_trans)
        x = np.power(10.0, x_trans - 3.0) - 0.01
        return x

    def __load_rho(self, path_scaler: str)->None:
        """
        Load the scalers for rho.
        
        Inputs:
        path_scaler: Path to the json file
        """
        # Open the file and assign the data to scaler_data object
        #with open(path_scaler, 'r') as file:
        #    scaler_data = json.load(file)

        # No files are needed

    def __transform_logg(self, x):
        x_trans = x
        return x_trans

    def __inverse_transform_logg(self, X_trans):
        x_trans = np.asarray(x_trans)
        x = x_trans
        return x
    
    def __load_logg(self, path_scaler: str)->None:
        """
        Load the scalers for logg.
        
        Inputs:
        path_scaler: Path to the json file
        """
        # Open the file and assign the data to scaler_data object
        #with open(path_scaler, 'r') as file:
        #    scaler_data = json.load(file)

        # No files are needed

    def __transform_radius(self, X):
        self.mean = np.mean(X)
        self.std = np.std(X)
        x_trans = (X - self.mean) / self.std
        return x_trans

    def __inverse_transform_radius(self, x_trans):
        x_trans = np.asarray(x_trans)
        x = np.multiply(x_trans, self.std) + self.mean
        return x

    def __load_radius(self, path_scaler: str)->None:
        """
        Load the scalers for radius
        
        Inputs:
        path_scaler: Path to the json file
        """
        # Open the file and assign the data to scaler_data object
        with open(path_scaler, 'r') as file:
            scaler_data = json.load(file)

        # Assign the stored elements to the corresponding properties
        self.mean = scaler_data['mean']
        self.std = scaler_data['std']

    def fit(self, x):
        pass

    def transform(self, x):
        x_trans = None
        self.mask = x > 0

        if self.param == 'T_eff':
            x_trans = self.__transform_teff(x)
        if self.param == 'Lum':
            x_trans = self.__transform_lum(x)
        if self.param == 'rho':
            x_trans = self.__transform_rho(x)
        if self.param == 'Mass':
            x_trans = self.__transform_mass(x)
        if self.param == 'logg':
            x_trans = self.__transform_logg(x)
        if self.param == 'Radius':
            x_trans = self.__transform_radius(x)

        x_trans = self.__apply_mask(x_trans)
        return x_trans

    def inverse_transform(self, x_trans):
        if self.param == 'T_eff':
            x = self.__inverse_transform_teff(x_trans)
        if self.param == 'Lum':
            x = self.__inverse_transform_lum(x_trans)
        if self.param == 'rho':
            x = self.__inverse_transform_rho(x_trans)
        if self.param == 'Mass':
            x = self.__inverse_transform_mass(x_trans)
        if self.param == 'logg':
            x = self.__inverse_transform_logg(x_trans)
        if self.param == 'Radius':
            x = self.__inverse_transform_radius(x_trans)

        # # Apply inverse masks to return to original values
        x = self.__apply_inverse_mask(x)

        return x

    def load_scaler(self, path_folder_scalers:str)->None:
        """
        Method to call the loader of each individual physical parameter.
        
        Inputs:
        scalers_path_folder: Path to the folder containing each individual scaler.
        """
        # Define the path to the specific json file
        path_scaler=join(path_folder_scalers, '.'.join([self.param, 'json']))
        # Load each scaler 
        if self.param == 'T_eff':
            self.__load_teff(path_scaler)
        if self.param == 'Lum':
            x = self.__load_lum(path_scaler)
        if self.param == 'rho':
            x = self.__load_rho(path_scaler)
        if self.param == 'Mass':
            x = self.__load_mass(path_scaler)
        if self.param == 'logg':
            x = self.__load_logg(path_scaler)
        if self.param == 'Radius':
            x = self.__load_radius(path_scaler)
    
    def __save(self, path_scaler:str, scaler:dict)->None:
        """
        Save the values for the mean and std of a scaler
        
        Inputs:
        path_scaler (str): Path to the json file
        scaler (dict): Dictionary containing the mean and std of the scaler.
        """
        with open(path_scaler, 'w') as file:
            json.dump(scaler, file)

    def save_scaler(self, path_folder_scalers:str)->None:
        """
        Method to call the saver of each individual physical parameter.
        
        Inputs:
        scalers_path_folder: Path containing the scalers.
        """
        # Define the path to the specific json file
        path_scaler=join(path_folder_scalers, '.'.join([self.param, 'json']))
        # Define the dictionary to save
        scaler = {'mean': self.mean, 'std':self.std}
        # Save the respective scaler
        self.__save(path_scaler, scaler)

    def __apply_mask(self, x_trans):
        x_trans = np.where(self.mask, x_trans, self.mask_value)
        return x_trans

    def __apply_inverse_mask(self, x):
        b = x > self.mask_value + 1
        x = np.where(b, x, -1.0)
        return x

