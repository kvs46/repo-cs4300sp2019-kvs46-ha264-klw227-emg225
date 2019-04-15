from . import *  
from app.irsystem.models.helpers import *
from app.irsystem.models.helpers import NumpyEncoder as NumpyEncoder

project_name = "Happy Campers"
net_id = "Katie Schretter: kvs46, Emily Gyles: emg226, Hanna Arfine: ha264, Kyra Wisniewski: klw227 "

@irsystem.route('/', methods=['GET'])
def search():
	query = request.args.get('search')
	if not query:
		data = []
		output_message = ''
		return render_template('search.html', name=project_name, netid=net_id, output_message=output_message, data=data)
	else:
		output_message = "Your search: " + query
		location = ['East Village', 'Chelsea']
		features = ['sunset', 'walk']
		results = [
			{'name': '875 Third Avenue', 
			'type':'POPS', 
			'area':'East Village', 
			'features':['Indoor', 'Plants', 'Walk'],
			'img': "static/images/city_background.png",
			'amenities': ['Artwork', 'Bicycle', 'Parking'],
			'rating': 4,
			'address': '875 third avenue',
			'summary': 'Four well-coordinated public spaces join forces to invigorate this full blockfront office tower on the east side of Third Avenue between East 52nd and 53rd Streets. An open space occupies the triangular area at the southeast corner of Third Avenue and East 53rd Street. A public circulation space inside the building connects users from the open space to the three-level covered pedestrian space. Miniscule arcade spaces cover the entrances to the covered pedestrian space at the northeast corner of Third Avenue and East 52nd Street and on East 53rd Street east of Third Avenue.'
			},

			{'name': '550 Madison Avenue', 
			'type':'POPS', 
			'area':'Chelsea', 
			'features':['Indoor', 'Seating', 'Food'],
			'img': "static/images/nyc.jpg",
			'amenities': ['Bathroom', 'Bicycle', 'Dogs']
			}
		]
		return render_template('results.html', loc_len = len(location), location=location, feat_len = len(features), features=features, results_len = len(results), results=results)
	# return render_template('results.html', data=data)
	


