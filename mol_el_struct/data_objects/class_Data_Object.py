
from utils.class_Disk_Jockey import Disk_Jockey
from utils.class_Journal import Journal, DisabledJournal



class Data_Object():

    # An instance of this class (or rather, any derived class) is used to communicate
    # with a single data object stored on disk.
    # A data object is a folder stored on disk, managed by a Disk Jockey instance

    main_path = "storage"
    object_type = None
    data_nodes = None

    def __init__(self, ID, j = None):

        cls = type(self) # the current class, is overridden by subclass

        if cls.object_type is None:
            raise NotImplementedError(f"{cls.__name__} missing property object_type")

        if cls.data_nodes is None:
            raise NotImplementedError(f"{cls.__name__} missing property data_nodes")

        self.ID = ID

        if j is None:
            self.j = DisabledJournal()
        else:
            self.j = j

        self.dj = Disk_Jockey(f"{Data_Object.main_path}/{cls.object_type}/{self.ID}", j) # Disk Jockey instance
        self.dj.create_data_nodes(cls.data_nodes)

        self.j.write(f"Initialised data object '{self.ID}' of the type '{cls.object_type}'", 5)

    def load_object(self, data_groups = None):
        self.dj.load_data(data_groups)

    def save_object(self, data_groups = None):
        self.dj.save_data(data_groups)

