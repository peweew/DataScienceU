#!/usr/bin/env python
# -*- coding: utf-8 -*-
import xml.etree.cElementTree as ET
import pprint
import re
import codecs
import json
"""
Your task is to wrangle the data and transform the shape of the data
into the model we mentioned earlier. The output should be a list of dictionaries
that look like this:

{
"id": "2406124091",
"type: "node",
"visible":"true",
"created": {
					"version":"2",
					"changeset":"17206049",
					"timestamp":"2013-08-03T16:43:42Z",
					"user":"linuxUser16",
					"uid":"1219059"
				},
"pos": [41.9757030, -87.6921867],
"address": {
					"housenumber": "5157",
					"postcode": "60625",
					"street": "North Lincoln Ave"
				},
"amenity": "restaurant",
"cuisine": "mexican",
"name": "La Cabana De Don Luis",
"phone": "1 (773)-271-5176"
}

You have to complete the function 'shape_element'.
We have provided a function that will parse the map file, and call the function with the element
as an argument. You should return a dictionary, containing the shaped data for that element.
We have also provided a way to save the data in a file, so that you could use
mongoimport later on to import the shaped data into MongoDB. 

Note that in this exercise we do not use the 'update street name' procedures
you worked on in the previous exercise. If you are using this code in your final
project, you are strongly encouraged to use the code from previous exercise to 
update the street names before you save them to JSON. 

In particular the following things should be done:
- you should process only 2 types of top level tags: "node" and "way"
- all attributes of "node" and "way" should be turned into regular key/value pairs, except:
		- attributes in the CREATED array should be added under a key "created"
		- attributes for latitude and longitude should be added to a "pos" array,
			for use in geospacial indexing. Make sure the values inside "pos" array are floats
			and not strings. 
- if the second level tag "k" value contains problematic characters, it should be ignored
- if the second level tag "k" value starts with "addr:", it should be added to a dictionary "address"
- if the second level tag "k" value does not start with "addr:", but contains ":", you can
	process it in a way that you feel is best. For example, you might split it into a two-level
	dictionary like with "addr:", or otherwise convert the ":" to create a valid key.
- if there is a second ":" that separates the type/direction of a street,
	the tag should be ignored, for example:

<tag k="addr:housenumber" v="5158"/>
<tag k="addr:street" v="North Lincoln Avenue"/>
<tag k="addr:street:name" v="Lincoln"/>
<tag k="addr:street:prefix" v="North"/>
<tag k="addr:street:type" v="Avenue"/>
<tag k="amenity" v="pharmacy"/>

	should be turned into:

{...
"address": {
		"housenumber": 5158,
		"street": "North Lincoln Avenue"
}
"amenity": "pharmacy",
...
}

- for "way" specifically:

	<nd ref="305896090"/>
	<nd ref="1719825889"/>

should be turned into
"node_refs": ["305896090", "1719825889"]
"""


lower = re.compile(r'^([a-z]|_)*$')
lower_colon = re.compile(r'^([a-z]|_)*:([a-z]|_)*$')
problemchars = re.compile(r'[=\+/&<>;\'"\?%#$@\,\. \t\r\n]')

CREATED = [ "version", "changeset", "timestamp", "user", "uid"]


def clean_postalcode(element):
	if len(element) == 6:
		return element[:3] + ' ' + element[3:]
	elif ';' in element:
		return element[: element.find(';')]
	elif 'AB' in element:
		return element[3:]
	else:
		return element
		
def clean_unit(element):
	if '#' in element:
		return element[1:]
	else:
		return element


def shape_element(element):
		node = {}
		if element.tag == "node" or element.tag == "way" :
				# YOUR CODE HERE  
				 
				createkeyset = ['version', 'changeset', 'timestamp', 'user', 'uid'] 
				poskeyset = ['lat', 'lon'] 
				
				for key in element.attrib.keys():   
					
					if key in createkeyset:
						if 'created' not in node.keys():
							node['created'] = {}
						node['created'][key] = element.get(key)
					elif key in poskeyset: 
						if 'pos' not in node.keys():
							node['pos'] = [0, 0]
							node['pos'][1] = float(element.get(key))
						else:
							node['pos'][0] = float(element.get(key))
					else:
						node[key] = element.get(key)
					
				for it in element.iter():  
					if it.tag == 'nd':
						if 'node_refs' not in node.keys():
							node['node_refs'] = []
						node['node_refs'].append(it.get('ref'))
					if it.tag == 'tag': 
						tag_types = it.get('k');
						mproblem = problemchars.search(tag_types)
						if mproblem:
							continue 
						elif 'addr' in tag_types: 
							if 'address' not in node.keys():
								node['address'] = {}
							
							realaddress = tag_types.split(':')
							address_type = realaddress[1]
							address_content = it.get('v') 
							
							if len(realaddress) == 2:
								if address_type == 'postcode':
									address_content = clean_postalcode(address_content)
								elif address_type == 'unit':
									address_content = clean_unit(address_content)
									
								node['address'][address_type] = address_content 
						else:
							node[tag_types] = it.get('v')
				
				node['type'] = element.tag
						 
				return node
		else:
				return None


def process_map(file_in, pretty = False):
		# You do not need to change this file
		file_out = "{0}.json".format(file_in[:file_in.find('.')])
		data = []
		with codecs.open(file_out, "w") as fo: 
				for _, element in ET.iterparse(file_in):
						el = shape_element(element)
						if el:
								data.append(el)
#								if pretty:
#										fo.write(json.dumps(el, indent=2)+"\n")
#								else:
#										fo.write(json.dumps(el) + "\n")
				
				fo.write(json.dumps(data, indent=2)+"\n")
		return data

def test():
		# NOTE: if you are running this code on your computer, with a larger dataset, 
		# call the process_map procedure with pretty=False. The pretty=True option adds 
		# additional spaces to the output, making it significantly larger.
		data = process_map('calgary.osm', True)
		#pprint.pprint(data)
		
	 
		print data[0]  

if __name__ == "__main__":
		test()