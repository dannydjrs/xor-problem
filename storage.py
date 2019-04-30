import pandas as pd

class Storage:

    def __init__(self, cols):
        """
        Construtor for the storage class.

        Parameters
        ----------
        cols : list
            List of the column names that will  be stored
        """
        self.cols = cols
        self.df = pd.DataFrame(columns=cols)

    def add_result(self, dict):
        """
        Adds a single row to the storage.

        Parameters
        ----------
        dict : dict
            Dictionary containing the row to store. Must have the same keys
            as set in constructor columns.

        Returns
        -------
        bool
            If the function was able to append the row to the dataframe successfully
        """
        if all(col in dict.keys() for col in self.cols):
            self.df = self.df.append(dict, ignore_index=True)
            return True
        else:
            return False

    def save_results(self, path):
        """
        Save results to a file

        Parameters
        ----------
        path : str
            The file path and name to store the results in
        """
        self.df.to_csv(path, index = None, header=True)
