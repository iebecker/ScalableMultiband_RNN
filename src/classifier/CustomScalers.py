import numpy as np


class ParamPhysScaler():

    def __init__(self,
                 param,
                 mask_value=0):
        self.param = param

        self.mask_value = mask_value

        self.mean = None
        self.std = None
        self.min = None

    def __transform_Teff(self, X):
        # X_trans = (X-3000.0)/1000.0
        self.mean = np.mean(X)
        self.std = np.std(X)
        X_trans = (X - self.mean) / self.std

        return X_trans

    def __inverse_transform_Teff(self, X_trans):
        X_trans = np.asarray(X_trans)
        # X = 1000.0*X_trans+3000.0
        X = np.multiply(X_trans, self.std) + self.mean
        return X

    def __transform_Lum(self, X):
        X_trans = np.log10(X + 0.01) + 4.0
        return X_trans

    def __inverse_transform_Lum(self, X_trans):
        X_trans = np.asarray(X_trans)
        X = np.power(10.0, X_trans - 4.0) - 0.01
        return X

    def __transform_Mass(self, X):
        X_trans = X
        return X_trans

    def __inverse_transform_Mass(self, X_trans):
        X_trans = np.asarray(X_trans)
        X = X_trans
        return X

    def __transform_rho(self, X):
        X_trans = np.log10(X + 0.01) + 3.0
        return X_trans

    def __inverse_transform_rho(self, X_trans):
        X_trans = np.asarray(X_trans)
        X = np.power(10.0, X_trans - 3.0) - 0.01
        return X

    def __transform_logg(self, X):
        X_trans = X
        return X_trans

    def __inverse_transform_logg(self, X_trans):
        X_trans = np.asarray(X_trans)
        X = X_trans
        return X

    def __transform_Radius(self, X):
        # X_trans = np.log10(X+0.01)+2.0
        self.mean = np.mean(X)
        self.std = np.std(X)
        X_trans = (X - self.mean) / self.std
        return X_trans

    def __inverse_transform_Radius(self, X_trans):
        X_trans = np.asarray(X_trans)
        # X = np.power(10.0, X_trans-2.0)-0.01
        X = np.multiply(X_trans, self.std) + self.mean
        return X

    def fit(self, X):
        pass

    def transform(self, X):
        X_trans = None
        self.mask = X > 0

        if self.param == 'T_eff':
            X_trans = self.__transform_Teff(X)
        if self.param == 'Lum':
            X_trans = self.__transform_Lum(X)
        if self.param == 'rho':
            X_trans = self.__transform_rho(X)
        if self.param == 'Mass':
            X_trans = self.__transform_Mass(X)
        if self.param == 'logg':
            X_trans = self.__transform_logg(X)
        if self.param == 'Radius':
            X_trans = self.__transform_Radius(X)

        X_trans = self.__apply_mask(X_trans)
        return X_trans

    def __apply_mask(self, X_trans):
        X_trans = np.where(self.mask, X_trans, self.mask_value)
        return X_trans

    def __apply_inverse_mask(self, X):
        b = X > self.mask_value + 1
        X = np.where(b, X, -1.0)
        return X

    def inverse_transform(self, X_trans):
        if self.param == 'T_eff':
            X = self.__inverse_transform_Teff(X_trans)
        if self.param == 'Lum':
            X = self.__inverse_transform_Lum(X_trans)
        if self.param == 'rho':
            X = self.__inverse_transform_rho(X_trans)
        if self.param == 'Mass':
            X = self.__inverse_transform_Mass(X_trans)
        if self.param == 'logg':
            X = self.__inverse_transform_logg(X_trans)
        if self.param == 'Radius':
            X = self.__inverse_transform_Radius(X_trans)

        # # Apply inverse masks to return to original values
        X = self.__apply_inverse_mask(X)

        return X
