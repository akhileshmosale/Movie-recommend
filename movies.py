import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

#fetch data and format it
data = fetch_movielens(min_rating=10.0)

#create model
model1 = LightFM(loss='bpr')
model2 = LightFM(loss='warp')

#train model
model1.fit(data['train'], epochs=30, num_threads=2)
model2.fit(data['train'], epochs=30, num_threads=2)


def recommendation(model1, model2, data, user_ids):

    #number of users and movies in training data
	n_users, n_items = data['train'].shape

    #generate recommendations for each user we input
	for user_id in user_ids:
		scores1 = model1.predict(user_id, np.arange(n_items))
		scores2 = model2.predict(user_id, np.arange(n_items))
	        top_items1 = data['item_labels'][np.argsort(-scores1)]
		top_items2 = data['item_labels'][np.argsort(-scores2)]

    #print out the results
        for x in top_items1[:5]:
		print("        %s" % x)
	for x in top_items2[:5]:
               print("        %s" % x)
            
recommendation(model1,model2, data, [3, 25, 450])
