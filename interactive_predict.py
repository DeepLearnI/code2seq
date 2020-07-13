from common import Common
from python_extractor.extractor import Extractor

SHOW_TOP_CONTEXTS = 10
MAX_PATH_LENGTH = 8
MAX_PATH_WIDTH = 2
# EXTRACTION_API = 'https://po3g2dx2qa.execute-api.us-east-1.amazonaws.com/production/extractmethods'


class InteractivePredictor:
    exit_keywords = ['exit', 'quit', 'q']

    def __init__(self, config, model):
        model.predict([])
        self.model = model
        self.config = config
        # self.path_extractor = Extractor(config, EXTRACTION_API, self.config.MAX_PATH_LENGTH, max_path_width=2)
        self.path_extractor = Extractor(self.config.MAX_PATH_LENGTH, MAX_PATH_WIDTH)

    @staticmethod
    def read_file(input_filename):
        with open(input_filename, 'r') as file:
            return file.readlines()

    def predict(self):
        input_filename = 'input.py'
        print('Serving')
        while True:
            print('Modify the file: "' + input_filename + '" and press any key when ready, or "q" / "exit" to exit')
            user_input = input()
            if user_input.lower() in self.exit_keywords:
                print('Exiting...')
                return
            try:
                predict_lines = list(path.strip() for path in self.path_extractor.extract_paths(input_filename))
                contexts = predict_lines[0].split()
                # space_padding = ' ' * (self.config.MAX_CONTEXTS - len(contexts) + 1)
                space_padding = ' ' * (200 - len(contexts) + 1)
                predict_lines[0] = ' '.join(contexts) + space_padding
            except ValueError as e:
                print(e)
                continue
            pc_info_dict = UnitDict()
            model_results = self.model.predict(predict_lines)

            prediction_results = Common.parse_results(model_results, pc_info_dict, topk=SHOW_TOP_CONTEXTS)
            for index, method_prediction in prediction_results.items():
                print('Original name:\t' + method_prediction.original_name)
                if self.config.BEAM_WIDTH == 0:
                    print('Predicted:\t%s' % [step.prediction for step in method_prediction.predictions])
                    for timestep, single_timestep_prediction in enumerate(method_prediction.predictions):
                        print('Attention:')
                        print('TIMESTEP: %d\t: %s' % (timestep, single_timestep_prediction.prediction))
                        for attention_obj in single_timestep_prediction.attention_paths:
                            print('%f\tcontext: %s,%s,%s' % (
                                attention_obj['score'], attention_obj['token1'], attention_obj['path'],
                                attention_obj['token2']))
                else:
                    print('Predicted:')
                    for predicted_seq in method_prediction.predictions:
                        print('\t%s' % predicted_seq.prediction)

class UnitDict(dict):

    def __getitem__(self, key):
        return key
