# What do I need to store?
#
# models(rl, deepl) -> file
# genetic action sets ->db?
# environments(field, spawn, rules) -> db? json?
# records(steps taken, result, food) -> db
#
# record model:
#   - player_id
#   - result
#   - steps_taken
#   - food_at_end
#   - env_type ('naive', 'extended')
#
# environment model:
#   - field_matrix (10x10 matrix of 0,1,2)
#   - spawn_coordinates (pair 0-9,0-9)?
#   - parameter_string (capital letter followed by integer depicting ratio: 'T40L30')
#   - env_type ('naive', 'extended')
#
# action set model:
#   - individual_id
#   - parent_id
#   - other_parent_id
#   - env_type ('naive', 'extended')
#   - action_set (list of tuples of 4+1 integers)

import pymongo
from utils.db_serializers import RecordSerializer, ScenarioSerializer, GeneticIndividualSerializer
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["records"] # genetic_individuals / scenarios / records

# record_col = db["records"]
# scenario_col = db["scenarios"]
# genetic_action_set_col = db["genetic_action_sets"]

def check():
    dblist = client.list_database_names()
    print(f"found databases: {dblist}")

    collist = db.list_collection_names()
    print(f"found collections: {collist}")

class DBContext:

    def __init__(self, collection, dt):
        """collection: collection name
        dt: 'r-s-g'"""
        self.collection = db[collection] #may need db source
        match dt:
            case 'r':
                self.serializer = RecordSerializer()
            case 's':
                self.serializer = ScenarioSerializer()
            case 'g':
                self.serializer = GeneticIndividualSerializer()

    def insert(self, obj):
        doc = self.serializer.serialize(obj)
        self.collection.insert_one(doc)
        print(f"inserted document into {self.collection}")

    def insert_many(self, objs):
        docs = self.serializer.serialize_many(objs)
        self.collection.insert_many(docs)
        print(f"inserted documents into {self.collection}")

    def get_all(self):
        docs = self.collection.find()
        print(f"queried documents from {self.collection}")
        return self.serializer.deserialize_many(docs)

    def drop(self):
        self.collection.drop()

def get_instance(collection: str, dt: str) -> DBContext:
    return DBContext(collection, dt)

check()










