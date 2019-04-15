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

		return render_template('results.html', loc_len = len(location), location=location, feat_len = len(features), features=features)
	# return render_template('results.html', data=data)
	


