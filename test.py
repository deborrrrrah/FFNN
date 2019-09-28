# batch = [row][input] ceritanya satu batch isi 3
'''
	input = [row][input]
	weights = [row][layer][input][output] -> input = node di layer dia, output -> node yang dituju
	
	
'''
input = [[3, 4, 5], [6, 4, 2], [2, 1, 6]]
weights = [[[10,20], [10, 20], [10, 20], [10, 20]], [[10], [10], [10]]]
n_features = 3
nb_nodes = 2
hidden_layer = 1
batch_size = 3

n_nodes = [n_features] + [nb_nodes] * hidden_layer + [1] # 1 for output

def forward_pass(input) :
	outputs = [] # initialize output to zero
	
	for row_idx in range(batch_size) : # iterate for each row
		row = [] # for output in each row
		
		print("start iterate layer")
		for layer_idx in range (hidden_layer + 1) : # iterate for each layer 
			layer = [] # for output in each layer 
			
			print("start iterate layer")
			for node_idx in range (n_nodes[layer_idx + 1]) : # iterate for each node in output layer
				node_v = 0
				
				print("start iterate node")
				node_v = weights[layer_idx][0][node_idx]
				for input_idx in range(n_nodes[layer_idx]) : # iterate for input node from input layer, +1 for bias
					node_v = node_v + (input[row_idx][input_idx] * weights[layer_idx][input_idx][node_idx])
					print(input_idx, " : ", node_v)
					
				
				print(node_v)
				layer.append(node_v)
			row.append(layer)
			
		outputs.append(row)
	return outputs
	
print(forward_pass(input))