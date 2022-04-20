# This script should implement the data preprocessor for the machine learning inference pipeline.
# There're two ways to input data to the preprocessor (considering a particular user's data):
# 1. A thread in the server extracts segments of data from the database and sends them to the preprocessor (somehow).
# 2. The preprocessor directly accesses data from the database.
#
# The output of the preprocessor must be passed to the feature extractor (that's working on the same user's data).
