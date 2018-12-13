import numpy as np

def load_surface_obj(file):

    vertices = []
    faces = []
    try:
        f = open(file)
        for line in f:
            if line[:2] == "v ":
                index1 = line.find(" ") + 1
                index2 = line.find(" ", index1 + 1)
                index3 = line.find(" ", index2 + 1)

                vertex = (float(line[index1:index2]), float(line[index2:index3]), float(line[index3:-1]))
                vertex = (round(vertex[0], 2), round(vertex[1], 2), round(vertex[2], 2))
                vertices.append(vertex)

            elif line[0] == "f":
                face = []
                tmp=line.split(' ')
                face.append(int(tmp[1].split("/")[0])-1)
                face.append(int(tmp[2].split("/")[0])-1)
                face.append(int(tmp[3].split("/")[0])-1)
                faces.append(face)

        f.close()
    except IOError:
        print(".obj file not found.")

    return [np.array(vertices), np.array(faces)]