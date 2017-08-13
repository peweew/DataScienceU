#!/usr/bin/env python
""" 
"""

def get_db(db_name):
	from pymongo import MongoClient
	client = MongoClient('localhost:27017')
	db = client[db_name]
	return db


def make_query():
	query = {}
	return query


def make_pipeline1():
	# complete the aggregation pipeline
	pipeline = [{"$match" : {"address.postcode" : {"$exists":1}}},
								{"$group" : {"_id" : "$address.postcode", "count" : {"$sum" : 1}}}, 
								{'$sort' : {'count' : -1}}]
							
	return pipeline
	
def make_pipeline2():
	# complete the aggregation pipeline
	pipeline = [{"$match":{"address.city":{"$exists":1}}}, 
							{"$group":{"_id" : "$address.city", "count" : {"$sum":1}}}, 
							{"$sort":{"count": -1}}]
							
	return pipeline
	

def aggregate(db, pipeline):
	
	return [doc for doc in db.calgarymap.aggregate(pipeline)]


if __name__ == '__main__':
	db = get_db('examples')  
	
	print '\nsize of file: {}'.format(db.calgarymap.find().count())
	
	print '\nnumber of nodes: {}'.format(db.calgarymap.find({"type":"node"}).count())
	print '\nnumber of ways: {}'.format(db.calgarymap.find({"type":"way"}).count())
	print '\nnumber of unique users: {}'.format(db.calgarymap.distinct("created.user").__len__())
	
	top_user = db.calgarymap.aggregate([{"$group":{"_id":"$created.user", "count":{"$sum":1}}}, {"$sort":{"count":-1}}, {"$limit":10}]) 
	print '\nThe top user: '
	total_count = 0
	for a in top_user:
			print a
			total_count += a['count']
	print total_count
			
	print '\nThe number of coffee store: {}'.format(db.calgarymap.find({'cuisine' : 'coffee_shop'}).count())
		
	 
	
	top_amenity =  db.calgarymap.aggregate([{"$match":{"amenity":{"$exists":1}}}, {"$group":{"_id":"$amenity",
	"count":{"$sum":1}}}, {"$sort":{"count":-1}}, {"$limit":10}])
	print '\nTop 10 amenity: ' 
	for a in top_amenity:
		print a
		
		
	top_cuisine =  db.calgarymap.aggregate([{"$match":{"cuisine":{"$exists":1}}}, {"$group":{"_id":"$cuisine",
		"count":{"$sum":1}}}, {"$sort":{"count":-1}}, {"$limit":10}])
	print '\nTop 10 cuisine: '
	for a in top_cuisine:
			print a
	
	pipeline = make_pipeline1()
	result_aggregate = aggregate(db, pipeline) 
	import pprint
	print '\nTop postal code:'
	#pprint.pprint(result_aggregate)














