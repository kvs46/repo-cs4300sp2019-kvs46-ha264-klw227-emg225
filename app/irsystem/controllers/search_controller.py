from . import *  
from app.irsystem.models.helpers import *
from app.irsystem.models.helpers import NumpyEncoder as NumpyEncoder
# from .prototype1 import main as get_results
# from .prototype1 import reviews_to_array
from .data_analysis import main as get_results_updated
from .data_analysis import good_types

project_name = "Happy Campers"
net_id = "Katie Schretter: kvs46, Emily Gyles: emg226, Hanna Arfine: ha264, Kyra Wisniewski: klw227 "

@irsystem.route('/', methods=['GET'])

def search():
	boroughs = request.args.get('boroughs')
	keywords = request.args.get('keyword')
	if not boroughs:
		goodtypes = good_types()
		keywords = ['dogs', 'new', 'space', 'sports', 'community', 'family', 'quiet', 'view', 'water', 'child-friendly', 'pretty']
		return render_template('map.html', keywords=keywords, len_words=len(keywords), goodtypes=goodtypes)
	if boroughs and keywords:
		location = boroughs.lower().split(",")
		features = keywords.lower().split(",")
		proto_results = get_results_updated(location, features)

		return render_template('results.html', loc_len = len(location), location=location, feat_len = len(features), features=features, proto_results=proto_results, len_results=len(proto_results))
	else:
		return render_template('map.html', keywords=keywords, len_words=len(keywords))
	# return render_template('results.html', data=data)
	


