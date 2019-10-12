from scvi.dataset import Dataset10X
import logging
import os
import pickle
import tarfile

import numpy as np
import pandas as pd
import scipy.io as sp_io
import shutil
from scipy.sparse import csr_matrix

from scvi.dataset.dataset import CellMeasurement

logger = logging.getLogger(__name__)

class Dataset10XCellTypes(Dataset10X):

    def __init__(self,dataset_name=None,metadata_file_name=None,**args):

        self.metadata_file_name = metadata_file_name

        super().__init__(
            dataset_name=dataset_name,
            **args
        )



    def populate(self):

        logger.info("Preprocessing dataset")

        was_extracted = False
        if len(self.filenames) > 0:
            file_path = os.path.join(self.save_path, self.filenames[0])
            if not os.path.exists(file_path[:-7]):  # nothing extracted yet
                if tarfile.is_tarfile(file_path):
                    logger.info("Extracting tar file")
                    tar = tarfile.open(file_path, "r:gz")
                    tar.extractall(path=self.save_path)
                    was_extracted = True
                    tar.close()

        # get exact path of the extract, for robustness to changes is the 10X storage logic
        path_to_data, suffix = self.find_path_to_data()

        # get filenames, according to 10X storage logic
        measurements_filename = "genes.tsv" if suffix == "" else "features.tsv.gz"
        barcode_filename = "barcodes.tsv" + suffix

        matrix_filename = "matrix.mtx" + suffix
        expression_data = sp_io.mmread(os.path.join(path_to_data, matrix_filename)).T
        if self.dense:
            expression_data = expression_data.A
        else:
            expression_data = csr_matrix(expression_data)

        labels = None
        cell_types = None

        # Cell-types
        current_dir = os.path.dirname(os.path.realpath(__file__))
        if self.metadata_file_name is not None:
            metadata = pickle.load(
                open(os.path.join(current_dir, self.metadata_file_name), "rb")
            )
            labels = np.array(metadata['clusters']).reshape(-1)
            cell_types = np.array(metadata['list_clusters']).reshape(-1)



        # group measurements by type (e.g gene, protein)
        # in case there are multiple measurements, e.g protein
        # they are indicated in the third column
        gene_expression_data = expression_data
        measurements_info = pd.read_csv(
            os.path.join(path_to_data, measurements_filename), sep="\t", header=None
        )
        Ys = None
        if measurements_info.shape[1] < 3:
            gene_names = measurements_info[self.measurement_names_column].astype(np.str)
        else:
            gene_names = None
            for measurement_type in np.unique(measurements_info[2]):
                # .values required to work with sparse matrices
                measurement_mask = (measurements_info[2] == measurement_type).values
                measurement_data = expression_data[:, measurement_mask]
                measurement_names = measurements_info[self.measurement_names_column][
                    measurement_mask
                ].astype(np.str)
                if measurement_type == "Gene Expression":
                    gene_expression_data = measurement_data
                    gene_names = measurement_names
                else:
                    Ys = [] if Ys is None else Ys
                    if measurement_type == "Antibody Capture":
                        measurement_type = "protein_expression"
                        columns_attr_name = "protein_names"
                        # protein counts are inherently not sparse
                        # protein counts are inherently not sparse
                        if self.dense is not True:
                            measurement_data = measurement_data.A
                    else:
                        measurement_type = measurement_type.lower().replace(" ", "_")
                        columns_attr_name = measurement_type + "_names"
                    measurement = CellMeasurement(
                        name=measurement_type,
                        data=measurement_data,
                        columns_attr_name=columns_attr_name,
                        columns=measurement_names,
                    )
                    Ys.append(measurement)
            if gene_names is None:
                raise ValueError(
                    "When loading measurements, no 'Gene Expression' category was found."
                )

        batch_indices, cell_attributes_dict = None, None
        if os.path.exists(os.path.join(path_to_data, barcode_filename)):
            barcodes = pd.read_csv(
                os.path.join(path_to_data, barcode_filename), sep="\t", header=None
            )
            cell_attributes_dict = {
                "barcodes": np.squeeze(np.asarray(barcodes, dtype=str))
            }
            # As of 07/01, 10X barcodes have format "%s-%d" where the digit is a batch index starting at 1
            batch_indices = np.asarray(
                [barcode.split("-")[-1] for barcode in cell_attributes_dict["barcodes"]]
            )
            batch_indices = batch_indices.astype(np.int64) - 1

        logger.info("Finished preprocessing dataset")

        self.populate_from_data(
            X=gene_expression_data,
            batch_indices=batch_indices,
            labels=labels,
            gene_names=gene_names,
            cell_types=cell_types,
            cell_attributes_dict=cell_attributes_dict,
            Ys=Ys,
        )
        self.filter_cells_by_count()

        # cleanup if required
        if was_extracted and self.remove_extracted_data:
            logger.info("Removing extracted data at {}".format(file_path[:-7]))
            shutil.rmtree(file_path[:-7])
