from . import *  
from app.irsystem.models.helpers import *
from app.irsystem.models.helpers import NumpyEncoder as NumpyEncoder
from .data_analysis import main as get_results_updated
from .data_analysis import good_types
from .data_analysis import names_array
from .data_analysis import similar_parks

project_name = "Urban Gems"
net_id = "Katie Schretter: kvs46, Emily Gyles: emg226, Hanna Arfine: ha264, Kyra Wisniewski: klw227 "

@irsystem.route('/', methods=['GET'])

def search():
	boroughs = request.args.get('boroughs')
	keywords = request.args.get('keyword')

	
	simto = request.args.get('simto')

	if not boroughs:
		goodtypes = good_types()
		keywords = ['dogs', 'new', 'space', 'sports', 'community', 'family', 'quiet', 'view', 'water', 'child-friendly', 'pretty']
		names = names_array()
		return render_template('map.html', keywords=keywords, len_words=len(keywords), goodtypes=goodtypes, parknames=names)
	if boroughs and keywords:
		location = boroughs
		features = keywords.lower().split(",")
		total_boros = ["queens", "manhattan", "staten island", "brooklyn", "bronx"]
		nlist = []
		for b in boroughs:
			if b not in total_boros:
				nlist.append(b)

		if simto:
			similar = similar_parks(simto)
			location = ['none']
			features = ['none']
			proto_results = similar
			return render_template('results.html', loc_len = len(location), location=location, feat_len = len(features), features=features, proto_results=proto_results, len_results=len(proto_results), simto=simto)
		else:
			proto_results = get_results_updated(location, features)

			return render_template('results.html', loc_len = len(location), location=location, feat_len = len(features), features=features, proto_results=proto_results, len_results=len(proto_results), nlist=nlist, nlen = len(nlist))
	else:
		return render_template('map.html', keywords=keywords, len_words=len(keywords))


