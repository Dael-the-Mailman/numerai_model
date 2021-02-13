import numerapi
NAPI = numerapi.NumerAPI(verbosity="info")

# Download new data
DIR = "E:/datasets/numerai_data"
NAPI.download_current_dataset(dest_path=DIR, unzip=True)

# # Load data
# full_path = f'{DIR}/numerai_dataset_{NAPI.get_current_round()}/'
# train = pd.read_csv(full_path + 'numerai_training_data.csv')
# test_df = pd.read_csv(full_path + 'numerai_tournament_data.csv')

# # Split validation and test
# val = test_df[test_df['data_type'] == 'validation']
# test = test_df[test_df['data_type'] != 'validation']