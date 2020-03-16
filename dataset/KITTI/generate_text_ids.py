views = [0,1,2,3,4]
views2 = [13,14,15,16]
folders = ["2","3"]
iterations = []
for i in range(200):
	iterations.append("{0:06}".format(i))

j=0
for i in iterations:
	for v in views:
		for f in folders:
			print(" ".join(["image_{folder}/{iteration}_0{view}.png".format(folder=f,iteration=i,view=v),
				"image_{folder}/{iteration}_0{view}.png".format(folder=f,iteration=i,view=v+1),
				"image_{folder}/{iteration}_0{view}.png".format(folder=f,iteration=i,view=v+2),
				"image_{folder}/{iteration}_0{view}.png".format(folder=f,iteration=i,view=v+3),
				"image_{folder}/{iteration}_0{view}.png".format(folder=f,iteration=i,view=v+4),
				"{0:06}".format(j)
				]))
			j+=1
	for v in views2:
		for f in folders:
			print(" ".join(["image_{folder}/{iteration}_{view}.png".format(folder=f,iteration=i,view=v),
				"image_{folder}/{iteration}_{view}.png".format(folder=f,iteration=i,view=v+1),
				"image_{folder}/{iteration}_{view}.png".format(folder=f,iteration=i,view=v+2),
				"image_{folder}/{iteration}_{view}.png".format(folder=f,iteration=i,view=v+3),
				"image_{folder}/{iteration}_{view}.png".format(folder=f,iteration=i,view=v+4),
				"{0:06}".format(j)
				]))
			j+=1
