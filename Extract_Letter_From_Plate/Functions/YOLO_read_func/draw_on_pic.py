#For input, enter the label file in text format




def get_coord(label_file, img_width, img_height):

    lfile = open(label_file)
    coords = []
    all_coords = []
    all_coords1 = []

    for line in lfile:
        l = line.split(" ")

        coords = list(map(float, list(map(float, l[1:5]))))
        x1 = float(img_width) * (2.0 * float(coords[0]) - float(coords[2])) / 2.0
        y1 = float(img_height) * (2.0 * float(coords[1]) - float(coords[3])) / 2.0
        x2 = float(img_width) * (2.0 * float(coords[0]) + float(coords[2])) / 2.0
        y2 = float(img_height) * (2.0 * float(coords[1]) + float(coords[3])) / 2.0

        int_x1 = int(x1)
        int_x2 = int(x2)
        int_y1 = int(y1)
        int_y2 = int(y2)

        tmp1 = (int_x1,int_y1)
        tmp2 = (int_x2,int_y2)
        tmp3 = [tmp1, tmp2]
        tmp = [x1, y1, x2, y2]
        all_coords.append(tmp3)

      all_coords1.append(list(map(int, tmp)))
      #all_coords is a list of tupples to be drawn back into the image
      #all_coords1 is a list of cordinates
    lfile.close()
    return all_coords, all_coords1

import cv2
coords, coords1 = get_coord('ac_301_b.txt', 987,987)
print(coords)
print(coords1)

#image file should be the one you would like to write your cordinates on
image = cv2.imread('ac_301_b.jpg')
thickness = 2
color = (0, 255, 0) # Green color
for dimensions in coords:
    print(dimensions)
    cv2.rectangle(image, dimensions[0], dimensions[1], color, thickness)

#print(image)
image = cv2.imwrite('imageWithRectCords.png', image)