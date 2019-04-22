from . import *  
from app.irsystem.models.helpers import *
from app.irsystem.models.helpers import NumpyEncoder as NumpyEncoder
from .prototype1 import main as get_results

project_name = "Happy Campers"
net_id = "Katie Schretter: kvs46, Emily Gyles: emg226, Hanna Arfine: ha264, Kyra Wisniewski: klw227 "

@irsystem.route('/', methods=['GET'])

def search():
	boroughs = request.args.get('boroughs')
	keywords = request.args.get('keyword')
	if not boroughs:
		keywords = ['dogs', 'new', 'space', 'sports', 'community', 'family', 'quiet', 'view', 'water', 'child-friendly', 'pretty']
		return render_template('map.html', keywords=keywords, len_words=len(keywords))

	else:
		location = boroughs.lower().split(",")
		features = keywords.lower().split(",")
		proto_results = get_results(location, features)
		
		return render_template('results.html', loc_len = len(location), location=location, feat_len = len(features), features=features, proto_results=proto_results, len_results=len(proto_results))
	# return render_template('results.html', data=data)
	


