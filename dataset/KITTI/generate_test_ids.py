views = [0]
folders = ["2","3"]
iterations = []
for i in range(200):
	iterations.append("{0:06}".format(i))

j=0
for i in iterations:
	print(" ".join([#"image_{folder}/{iteration}_0{view}.png".format(folder=f,iteration=i,view=v),
		"image_2/{iteration}_09.png".format(iteration=i),
		"image_2/{iteration}_10.png".format(iteration=i),
		"image_2/{iteration}_11.png".format(iteration=i),
		"flow_noc/{iteration}_10.png".format(iteration=i),
		"flow_occ/{iteration}_10.png".format(iteration=i),
		#"image_{folder}/{iteration}_0{view}.png".format(folder=f,iteration=i,view=v+4),
		"{0:06}".format(j)
		]))
	j+=1