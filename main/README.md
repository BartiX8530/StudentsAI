AI for analyzing student data and performance (and not only that!)

Almost all of the code is documented through comments, for a simple guide:

main.py - performs feature selection on a simpler model (linear regression), trains a model, analyzes it's performance through permutation importance, and gives options to save it and analyze it with SHAP
shap_analyze.py - analyzes a finished model using SHAP (you need to manually edit the used features)
gridSearch.py - performs a gridSearch for the best performing model
test.py - compares the neural network model with baseline models (linear regression and decision tree)
checked models.py - other models that have been checked for results (not expanded upon too much)
modelBuildUtil.py - used in other files to build a model with parameters specified in a directionary

There is a finished model included, with quite good performance
