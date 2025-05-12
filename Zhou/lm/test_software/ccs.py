from ucimlrepo import fetch_ucirepo 
# fetch dataset 
concrete_compressive_strength = fetch_ucirepo(id=165) 
# data (as pandas dataframes) 
X = concrete_compressive_strength.data.features 
y = concrete_compressive_strength.data.targets 
  
# metadata 
print(concrete_compressive_strength.metadata) 
  
# variable information 
print(concrete_compressive_strength.variables) 