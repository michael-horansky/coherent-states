# -----------------------------------------------------------------------------
# -------------------------------- Disk Jockey --------------------------------
# -----------------------------------------------------------------------------
# Data storage manager.
# Data is organised into data groups, where each data group contains a set of
# datafiles. Every datafile can have a corresponding metadata file.

from pathlib import Path
import csv
import base64
import json
import numpy as np
import pickle


# JSON serialization of complex ndarrays
# Solution by hpaulj https://stackoverflow.com/a/24375113/901925


# Overload dump/load to default use this behavior.
def dumps(datum, **kwargs):
    return json.dumps(pickle.dumps(datum).decode('latin-1'), **kwargs)

def loads(string, **kwargs):
    return pickle.loads(json.loads(string, **kwargs).encode('latin-1'))

def dump(*args, **kwargs):
    return json.dump(*args, **kwargs)

def load(*args, **kwargs):
    return json.load(*args, **kwargs)

class Disk_Jockey():

    object_type_write_mode = {
            "txt" : "w",
            "csv" : "w",
            "json" : "w",
            "pkl" : "wb"
        }
    object_type_read_mode = {
            "txt" : "r",
            "csv" : "r",
            "json" : "r",
            "pkl" : "rb"
        }
    force_pickled_meta = ["csv"] # file types in this list require their metadata files to be pickled
    columnwise_data_objects = ["csv"]

    def __init__(self, storage_path, journal_instance = None):
        self.storage_path = storage_path
        self.journal_instance = journal_instance # Journal, if provided
        self.data_nodes = {} # [data_group] = datum name
        self.is_data_initialised = {} # [data_group][datum_name] = was anything committed to the datum?
        self.data_bulks = {} # [data_group][datum_name] = some object
        self.data_bulk_types = {} # [data_group][datum_name] = how to interpret data object. Typically "csv" for structured datapoints
        self.metadata = {} # [data_group][datum_name] = {"metadatum" : ..., "metadatum" : ...}. Has to be JSON compatible
        self.metadata_types = {} # [data_group][datum_name] = metadata file extension. pkl if necessary; otherwise json

        self.column_datatypes = {} # For column-wise object types, this keeps the track of datatypes for each object. [data_group][header key] = type

        self.did_load_from_disk = False # Was data ever loaded from the disk?

        # Check if subdirectory exists
        main_dir = Path(f"{self.storage_path}")
        if not main_dir.exists():
            self.did_load_from_disk = True # There's nothing to load!

        # Create subdirectory if not exists
        Path(f"{self.storage_path}").mkdir(parents=True, exist_ok=True)

        # Maybe also have data_group_metadata, with one entry per data group? So the user doesn't have to remember what each data group is

    def make_dir_name_unique(self):
        # Appends "_ver_{n}" until not existing on disk
        n = 2
        while(True):
            proposed_dir_path = f"{self.storage_path}_ver_{n}"
            main_dir = Path(proposed_dir_path)
            if not main_dir.exists():
                self.did_load_from_disk = True # There's nothing to load!
                self.storage_path = proposed_dir_path
                # We need to create the node subdirectories
                main_dir.mkdir(parents=True, exist_ok=True)
                for data_group in self.data_nodes.keys():
                    Path(f"{self.storage_path}/{data_group}").mkdir(parents=True, exist_ok=True)
                break
            n += 1


    def create_data_nodes(self, data_nodes):
        # data_nodes[data_group][datum_name] = datum_type
        for data_group, data_setups in data_nodes.items():
            if data_group not in self.data_nodes.keys():
                # We add new data_group to all keys
                self.data_nodes[data_group] = []
                self.is_data_initialised[data_group] = {}
                self.data_bulks[data_group] = {}
                self.data_bulk_types[data_group] = {}
                self.metadata[data_group] = {}
                self.metadata_types[data_group] = {}
                self.column_datatypes[data_group] = {}

                # We create the proper subdirectory if not exists
                Path(f"{self.storage_path}/{data_group}").mkdir(parents=True, exist_ok=True)

            for datum_name, datum_type in data_setups.items():
                if datum_name not in self.data_nodes[data_group]:
                    self.data_nodes[data_group].append(datum_name)
                    self.is_data_initialised[data_group][datum_name] = False
                    self.data_bulks[data_group][datum_name] = None
                    self.data_bulk_types[data_group][datum_name] = datum_type
                    self.metadata[data_group][datum_name] = {}
                    self.metadata_types[data_group][datum_name] = "json"
                    if datum_type in Disk_Jockey.force_pickled_meta:
                        self.metadata_types[data_group][datum_name] = "pkl"

    def commit_datum_bulk(self, data_group, datum_name, datum_bulk, header_row = False):
        # Objects expected for given data types:
        #     "csv" : A list of lists, interpreted as a list of rows, or a list of dicts, with keys being the header items.
        #     "txt" : A string.
        #     "json" : Any object which can be dumped
        self.data_bulks[data_group][datum_name] = datum_bulk
        self.is_data_initialised[data_group][datum_name] = True
        if self.data_bulk_types[data_group][datum_name] in Disk_Jockey.columnwise_data_objects:
            # We store the column datatypes
            self.column_datatypes[data_group][datum_name] = {}
            if header_row:
                for i in range(len(datum_bulk[1])):
                    self.column_datatypes[data_group][datum_name][datum_bulk[0][i]] = type(datum_bulk[1][i])
            else:
                for column_key in datum_bulk[0].keys():
                    self.column_datatypes[data_group][datum_name][column_key] = type(datum_bulk[0][column_key])

    def commit_metadatum(self, data_group, datum_name, metadatum):
        self.metadata[data_group][datum_name] = metadatum

    def commit_metadatum_point(self, data_group, datum_name, metadatum_point_name, metadatum_point_value):
        self.metadata[data_group][datum_name][metadatum_point_name] = metadatum_point_value

    def set_metadatum_type(self, data_group, datum_name, metadatum_type):
        self.metadata_types[data_group][datum_name] = metadatum_type

    def datum_directory(self, data_group, datum_name):
        # Returns the datum directory as a path
        return(f"{self.storage_path}/{data_group}/{datum_name}.{self.data_bulk_types[data_group][datum_name]}")

    def metadata_directory(self, data_group, datum_name):
        # Returns the directory to the metadata for the specific datum as a path
        return(f"{self.storage_path}/{data_group}/{datum_name}_meta.{self.metadata_types[data_group][datum_name]}")

    def save_datum(self, data_group, datum_name):
        # data_group specifies subdirectory name.
        # datum_name specifies filename

        if (not self.did_load_from_disk):
            response = self.journal_instance.ask_yes_no("WARNING: Saving to an existing data directory. Data may be owerwritten. Proceed?")
            #print("WARNING: Saving to an existing data directory. Data may be owerwritten. Proceed? (y/n)")
            if response == False:
                # Do not proceed--instead, we alter storage path to a unique string
                self.make_dir_name_unique()
            else:
                # Proceed and mark loading as safe
                self.did_load_from_disk = True

        if self.is_data_initialised[data_group][datum_name]:
            # datum bulk
            datum_bulk_file = open(self.datum_directory(data_group, datum_name), Disk_Jockey.object_type_write_mode[self.data_bulk_types[data_group][datum_name]])
            if self.data_bulk_types[data_group][datum_name] == "txt":
                datum_bulk_file.write(self.data_bulks[data_group][datum_name])
            elif self.data_bulk_types[data_group][datum_name] == "csv":
                # As I said: either a list of lists, or a list of dicts

                col_keys = self.column_datatypes[data_group][datum_name].keys()

                if isinstance(self.data_bulks[data_group][datum_name][0], dict):
                    # list of dicts
                    datum_bulk_writer = csv.DictWriter(datum_bulk_file, fieldnames = col_keys)
                    datum_bulk_writer.writeheader()
                    datum_bulk_writer.writerows(self.data_bulks[data_group][datum_name])
                else:
                    # List of lists. Does it contain a header?
                    datum_bulk_writer = csv.writer(datum_bulk_file)

                    is_there_header_row = True
                    for col_key in col_keys:
                        if col_key not in self.data_bulks[data_group][datum_name][0]:
                            is_there_header_row = False
                            break
                    if not is_there_header_row:
                        datum_bulk_writer.writerow(col_keys)

                    datum_bulk_writer.writerows(self.data_bulks[data_group][datum_name])
            elif self.data_bulk_types[data_group][datum_name] == "json":
                json.dump(self.data_bulks[data_group][datum_name], datum_bulk_file, indent=2)
            elif self.data_bulk_types[data_group][datum_name] == "pkl":
                pickle.dump(self.data_bulks[data_group][datum_name], datum_bulk_file)

            datum_bulk_file.close()

            # metadatum
            if self.data_bulk_types[data_group][datum_name] in Disk_Jockey.columnwise_data_objects:
                # We also store the column datatypes
                self.commit_metadatum_point(data_group, datum_name, "column_datatypes", self.column_datatypes[data_group][datum_name])
            if not (self.metadata[data_group][datum_name] is None or self.metadata[data_group][datum_name] == {}):
                metadatum_file = open(self.metadata_directory(data_group, datum_name), Disk_Jockey.object_type_write_mode[self.metadata_types[data_group][datum_name]])
                if self.metadata_types[data_group][datum_name] == "pkl":
                    pickle.dump(self.metadata[data_group][datum_name], metadatum_file)
                elif self.metadata_types[data_group][datum_name] == "json":
                    json.dump(self.metadata[data_group][datum_name], metadatum_file, indent = 2)
                metadatum_file.close()

    def load_datum(self, data_group, datum_name):
        self.did_load_from_disk = True

        if self.is_data_initialised[data_group][datum_name]:
            self.journal_instance.write(f"WARNING: Overwriting an initialised data node at '{data_group}/{datum_name}' with data from disk.")
        # metadata
        metadatum_path = Path(self.metadata_directory(data_group, datum_name))
        if metadatum_path.is_file():
            metadatum_file = open(self.metadata_directory(data_group, datum_name), Disk_Jockey.object_type_read_mode[self.metadata_types[data_group][datum_name]])
            if self.metadata_types[data_group][datum_name] == "pkl":
                self.commit_metadatum(data_group, datum_name, pickle.load(metadatum_file))
            elif self.metadata_types[data_group][datum_name] == "json":
                self.commit_metadatum(data_group, datum_name, json.load(metadatum_file))
            metadatum_file.close()

            if "column_datatypes" in self.metadata[data_group][datum_name].keys():
                self.column_datatypes[data_group][datum_name] = self.metadata[data_group][datum_name]["column_datatypes"]
        datum_bulk_path = Path(self.datum_directory(data_group, datum_name))
        if datum_bulk_path.is_file():
            datum_bulk_file = open(self.datum_directory(data_group, datum_name), Disk_Jockey.object_type_read_mode[self.data_bulk_types[data_group][datum_name]])
            if self.data_bulk_types[data_group][datum_name] == "txt":
                self.commit_datum_bulk(data_group, datum_name, datum_bulk_file.read())
            elif self.data_bulk_types[data_group][datum_name] == "csv":
                datum_bulk_reader = csv.DictReader(datum_bulk_file, delimiter=',', quotechar='"')
                datum_bulk_rows = []
                for row in datum_bulk_reader:
                    sanitised_row = {}
                    for key, obj in row.items():
                        sanitised_row[key] = self.column_datatypes[data_group][datum_name][key](obj)
                    datum_bulk_rows.append(sanitised_row)
                self.commit_datum_bulk(data_group, datum_name, datum_bulk_rows)
            elif self.data_bulk_types[data_group][datum_name] == "json":
                self.commit_datum_bulk(data_group, datum_name, json.load(datum_bulk_file))
            elif self.data_bulk_types[data_group][datum_name] == "pkl":
                self.commit_datum_bulk(data_group, datum_name, pickle.load(datum_bulk_file))

            datum_bulk_file.close()
        else:
            self.journal_instance.write(f"WARNING: Failed to load node at '{data_group}/{datum_name}' from disk: No such file.")

    def save_root_metadata(self):
        root_metadata = {
            "data_nodes" : self.data_nodes,
            "data_bulk_types" : self.data_bulk_types,
            "metadata_types" : self.metadata_types
            }

        metadatum_file = open(f"{self.storage_path}/root_meta.json", Disk_Jockey.object_type_write_mode["json"])
        json.dump(root_metadata, metadatum_file, indent=2)
        metadatum_file.close()


    def save_data(self, data_groups = None):
        # Saves all initialised data to disk
        # If data_groups is None, saves all data. If it is a list of data_group
        # names, only saves data from those data groups.
        if data_groups is None:
            self.save_data(list(self.data_nodes.keys()))
        else:
            for data_group in data_groups:
                for datum_name in self.data_nodes[data_group]:
                    self.save_datum(data_group, datum_name)

        # Now we store the disk jockey metadata themselves, namely the node structure
        self.save_root_metadata()

    def load_root_metadata(self):
        metadatum_file = open(f"{self.storage_path}/root_meta.json", Disk_Jockey.object_type_read_mode["json"])
        loaded_root_metadata = json.load(metadatum_file)
        metadatum_file.close()

        # We add data nodes which are not committed yet but exist on disk
        self.create_data_nodes(loaded_root_metadata["data_bulk_types"])
        # We make sure loaded metadata types agree
        for data_group in loaded_root_metadata["metadata_types"].keys():
            for datum_name, metadata_type in loaded_root_metadata["metadata_types"][data_group].items():
                self.set_metadatum_type(data_group, datum_name, metadata_type)


    def load_data(self, data_groups = None):
        # Loads all data from disk
        # If data_groups is None, loads all data. If it is a list of data_group
        # names, only loads data from those data groups.


        if data_groups is None:
            self.load_data(list(self.data_nodes.keys()))
        else:
            # First, load the metadata as they exist on disk, so we know the proper types etc
            self.load_root_metadata()
            for data_group in data_groups:
                for datum_name in self.data_nodes[data_group]:
                    self.load_datum(data_group, datum_name)

