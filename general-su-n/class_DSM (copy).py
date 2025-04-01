# -----------------------------------------------------------------------------
# ------------------------------------ DSM ------------------------------------
# -----------------------------------------------------------------------------
# DSM - Data Storage Manager. Every data object (i.e. basis sample, wavef prop,
# Fock solution...) gets a special datum in DSM, which will be written into two
# files: the main data bulk and the metadata.

from pathlib import Path
import csv
#import json
import pyon
import numpy as np

class DSM():

    def __init__(self, storage_path):
        self.storage_path = storage_path
        self.data_nodes = {} # [data_group] = data name
        self.is_data_initialised = {} # [datum_name] = True or False
        self.data_bulks = {} # [datum_name] = some object
        self.data_bulk_types = {} # [datum_name] = how to interpret data object. Typically "csv" for structured datapoints
        self.metadata = {} # [datum_name] = {"metadatum" : ..., "metadatum" : ...}. Has to be JSON compatible


    def create_data_nodes(self, data_nodes):
        # data_nodes[data_group][datum_name] = datum_type
        for data_group, data_setups in data_nodes.items():
            if data_group not in self.data_nodes.keys():
                self.data_nodes[data_group] = []
            for datum_name, datum_type in data_setups.items():
                self.data_nodes[data_group].append(datum_name)
                self.is_data_initialised[datum_name] = False
                self.data_bulks[datum_name] = None
                self.data_bulk_types[datum_name] = datum_type
                self.metadata[datum_name] = {}

    def commit_datum_bulk(self, datum_name, datum_bulk):
        # Objects expected for given data types:
        #     "csv" : A list of lists, interpreted as a list of rows, or a list of dicts, with keys being the header items.
        #     "txt" : A string.
        #     "pyon" : Any object which can be dumped
        self.data_bulks[datum_name] = datum_bulk
        self.is_data_initialised[datum_name] = True

    def commit_metadatum(self, datum_name, metadatum):
        self.metadata[datum_name] = metadatum

    def commit_metadatum_point(self, datum_name, metadatum_point_name, metadatum_point_value):
        self.metadata[datum_name][metadatum_point_name] = metadatum_point_value

    """def sanitise_object_for_serialisation(self, obj):
        if isinstance(obj, np.ndarray):
    """

    def save_datum(self, datum_name):
        if self.is_data_initialised[datum_name]:
            # datum bulk
            datum_bulk_file = open(f"{self.storage_path}/{datum_name}.{self.data_bulk_types[datum_name]}", "w")
            if self.data_bulk_types[datum_name] == "txt":
                datum_bulk_file.write(self.data_bulks[datum_name])
            elif self.data_bulk_types[datum_name] == "csv":
                datum_bulk_writer = csv.writer(datum_bulk_file)
                datum_bulk_writer.writerows(self.data_bulks[datum_name])
            elif self.data_bulk_types[datum_name] == "pyon":
                datum_bulk_file.write(pyon.encode(self.data_bulks[datum_name]))
                #json.dump(self.sanitise_object_for_serialisation(self.data_bulks[datum_name]), datum_bulk_file, indent = 4)

            datum_bulk_file.close()

            # metadatum
            if not (self.metadata[datum_name] is None or self.metadata[datum_name] == {}):
                #metadatum_file = open(f"{self.storage_path}/{datum_name}_meta.pyon", "w")
                #json.dump(self.sanitise_object_for_serialisation(self.metadata[datum_name]), metadatum_file)

                #metadatum_file.close()
                pyon.to_file(self.metadata[datum_name], f"{self.storage_path}/{datum_name}_meta.pyon")

    def load_datum(self, datum_name):
        if self.is_data_initialised[datum_name]:
            print("  WARNING: Overwriting an initialised data node with data from disk.")
        datum_bulk_path = Path(f"{self.storage_path}/{datum_name}.{self.data_bulk_types[datum_name]}")
        if datum_bulk_path.is_file():
            datum_bulk_file = open(f"{self.storage_path}/{datum_name}.{self.data_bulk_types[datum_name]}", "r")
            if self.data_bulk_types[datum_name] == "txt":
                self.commit_datum_bulk(datum_name, datum_bulk_file.read())
            elif self.data_bulk_types[datum_name] == "csv":
                datum_bulk_reader = csv.DictReader(datum_bulk_file, delimiter=',', quotechar='"')
                datum_bulk_rows = []
                for row in datum_bulk_reader:
                    datum_bulk_rows.append(row)
                self.commit_datum_bulk(datum_name, datum_bulk_rows)
            elif self.data_bulk_types[datum_name] == "pyon":
                self.commit_datum_bulk(datum_name, pyon.decode(datum_bulk_file.read()))

            datum_bulk_file.close()

            # metadata
            metadatum_path = Path(f"{self.storage_path}/{datum_name}_meta.pyon")
            if metadatum_path.is_file():
                #metadatum_file = open(f"{self.storage_path}/{datum_name}_meta.json", "r")
                #self.commit_metadatum(datum_name, json.load(metadatum_file))
                #metadatum_file.close()
                self.commit_metadatum(datum_name, pyon.from_file(f"{self.storage_path}/{datum_name}_meta.pyon"))

    def save_data(self, data_groups = None):
        # Saves all initialised data to disk
        # If data_groups is None, saves all data. If it is a list of data_group
        # names, only saves data from those data groups.
        if data_groups is None:
            self.save_data(list(self.data_nodes.keys()))
        else:
            # Create subfolder if not exists
            Path(f"{self.storage_path}").mkdir(parents=True, exist_ok=True)

            for data_group in data_groups:
                for datum_name in self.data_nodes[data_group]:
                    self.save_datum(datum_name)

    def load_data(self, data_groups = None):
        # Loads all data from disk
        # If data_groups is None, loads all data. If it is a list of data_group
        # names, only loads data from those data groups.
        if data_groups is None:
            self.load_data(list(self.data_nodes.keys()))
        else:
            for data_group in data_groups:
                for datum_name in self.data_nodes[data_group]:
                    self.load_datum(datum_name)

