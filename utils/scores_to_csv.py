from db_entities import RecordModel, GameResult
from db_serializers import RecordSerializer
import pymongo
import numpy

client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client['records_gen_ex_selected'] #changed!

def main():
    with open("records_gen_ex_selected.csv", 'w') as f:
        collections = db.list_collection_names()
        for collection in sorted(collections):
            current_collection = db[collection]
            documents = current_collection.find()
            models: list[RecordModel] = RecordSerializer().deserialize_many(documents)
            scores = numpy.array([m.steps_taken for m in models])
            av = numpy.average(scores)
            med = numpy.median(scores)
            first = numpy.quantile(scores, 0.25)
            third = numpy.quantile(scores, 0.75)
            var = numpy.var(scores)
            dev = numpy.sqrt(var)
            ran = numpy.ptp(scores)
            num = len(scores)
            eaten = 0
            won = 0
            oot = 0
            shot_lions = 0
            for record in models:
                if record.result == "EATEN_BY_LION":
                    eaten += 1
                if record.result == "COMPLETE":
                    won += 1
                if record.result == "OOT":
                    oot += 1
                shot_lions += record.shot_lions
            shot_lions /= len(models) #avg shot lions
            line = f"{','.join(collection.split(';'))},{av:.2f},{med},{first},{third},{num},{var:.2f},{dev:.2f},{ran},{eaten},{won},{oot},{shot_lions:.2f}"
            f.write(line + "\n")

if __name__ == '__main__':
    main()


