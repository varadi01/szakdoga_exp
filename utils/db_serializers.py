from utils.db_entities import RecordModel, ScenarioModel, GeneticIndividualModel
from game_environment.scenario import TileState, Step, Environment
from utils.genetic_action_utils import Action, ActionHolder

class Serializer:

    def serialize(self, obj):
        pass

    def deserialize(self, doc):
        pass

    def serialize_many(self, iterable):
        docs = []
        for _ in iterable:
            if type(_) == dict:
                continue #dunno
            docs.append(self.serialize(_))
        return docs

    def deserialize_many(self, iterable):
        objs = []
        for _ in iterable:
            objs.append(self.deserialize(_))
        return objs

class RecordSerializer(Serializer):

    def serialize(self, obj: RecordModel) -> dict:
        return {
            "player_id": obj.player_id,
            "result": obj.result.value,
            "steps_taken": obj.steps_taken,
            "food_at_end": obj.food_at_end,
            "env_type": obj.env_type,
            "env_parameter_string": obj.env_parameter_string
        }


    def deserialize(self, doc) -> RecordModel:
        return RecordModel(
            doc["player_id"],
            doc["result"], #RecordModel.GameResult(doc["result"]),
            doc["steps_taken"],
            doc["food_at_end"],
            doc["env_type"],
            doc["env_parameter_string"]
        )

class ScenarioSerializer(Serializer):

    def serialize(self, obj: ScenarioModel) -> dict:
        return {
            "board": obj.board,
            "spawn_coordinates": list(obj.spawn),
            "parameter_string": obj.parameter_string,
            "env_type": obj.env_type
        }

    def deserialize(self, doc) -> ScenarioModel:
        return ScenarioModel(
            doc["board"], #sus
            doc["spawn_coordinates"],
            doc["parameter_string"],
            doc["env_type"]
        )

class GeneticIndividualSerializer(Serializer):

    def serialize(self, obj: GeneticIndividualModel) -> dict:
        return {
            "individual_id": obj.ind_id,
            "action_set": GeneticIndividualSerializer._action_set_serializer(obj.action_set),
            "env_type": obj.env_type,
            "parent_id": obj.parent_id,
            "other_parent_id": obj.other_parent_id
        }

    def deserialize(self, doc) -> GeneticIndividualModel:
        return GeneticIndividualModel(
            doc["individual_id"],
            GeneticIndividualSerializer._action_set_deserializer(doc["action_set"]),
            doc["env_type"],
            doc["parent_id":],
            doc["other_parent_id"]
        )
    @staticmethod
    def _action_set_serializer(action_set):
        new = []
        for action in action_set.actions:
            a_rep = action.env.get_as_list().extend(action.step.value)
            new.append(a_rep)
        return new

    @staticmethod
    def _action_set_deserializer(action_set):
        new = ActionHolder()
        for action in action_set:
            new.add_action(
                Action(
                    Environment.get_from_list(action[:4]),
                    Step(action[4])
                )
            )
        return new
