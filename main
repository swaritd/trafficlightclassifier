
# This function should take in RGB image input
# Analyze that image using your feature creation code and output a one-hot encoded label

#boolean flag if hot encoder is double encoded
def double_encoded(one_hot_encoded):
    if (np.sum(one_hot_encoded) > 1):
        return True
    else:
        return False
    
# boolean flag if hot encoder has a label
def has_label(one_hot_encoded):
    if (np.sum(one_hot_encoded) > 0):
        return True
    else:
        return False

# hot encode the image based on where the mean of the feature is in relation to the height of the
# traffic light.
def hot_encode_height(mean):
    # bin ranges
    # red = 0, 15
    # yellow = 10, 23
    # green = 20, 32
    one_hot_encoded = [0, 0, 0]
    
    if mean >= 0 and mean < 15:
        one_hot_encoded[0] = 1
    if mean >= 10 and mean < 23:
        one_hot_encoded[1] = 1
    if mean >= 20 and mean <= 32:
        one_hot_encoded[2] = 1
    
    return one_hot_encoded

def estimate_value(rgb_image):
    
    feature = feature_value(rgb_image)
    
    # get mean and bimodal boolean
    max_list2 = max_idx_rank(feature)
    mean = max_list2[0]
    bimodal = is_bimodal(max_list2, feature)
    
    one_hot_encoded = hot_encode_height(mean)
    return one_hot_encoded, bimodal 


def estimate_hueXvalue(rgb_image):
    
    feature = feature_valueXHue(rgb_image)
    
    # get mean and bimodal boolean
    max_list2 = max_idx_rank(feature)
    mean = max_list2[0]
    bimodal = is_bimodal(max_list2, feature)
    
    one_hot_encoded = hot_encode_height(mean)
    return one_hot_encoded, bimodal 
        
    
def estimate_color(rgb_image):
    feature = feature_rgb(rgb_image)
    
    one_hot_encoded = [0, 0, 0]
    # sum channels representing each light color
    red_sum = np.sum(feature[:,:,0])
    green_sum = np.sum(feature[:,:,1])
    yellow_sum = np.sum(feature[:,:,2])
    
    # one hot encode the color who has the greatest sum
    if red_sum > (yellow_sum + green_sum):
        one_hot_encoded[0] = 1
    if yellow_sum > (green_sum + red_sum):
        one_hot_encoded[1] = 1
    if green_sum > (yellow_sum + red_sum):
        one_hot_encoded[2] = 1
        
    return one_hot_encoded
        
# rules
    # * if all hues detected, or white is predominant. color is most liekly yellow.
    # * if a bimodal distribution of value vs traffic light length is found.
    #     - compute hue * value.
    #     - use hue * value mean retrieved from this plot.

def estimate_label(rgb_image, print_stats = False):
    
    ## TODO: Extract feature(s) from the RGB image and use those features to
    ## classify the image and output a one-hot encoded label
    predicted_label = [0,0,0]
    predicted_label_height = []
    predicted_label_color = []
    bimodal = False
    # copy picture
    image = np.copy(rgb_image)
    
    # get hot encode for value vs height
    predicted_label_Value, bimodal_value = estimate_value(image)
    
    if print_stats:
        print('height: {}, bimodal: {}'.format(predicted_label_Value, bimodal_value))
    
    # if bimodal compute hue * value and get the hot encode vs height
    if bimodal == True:
        predicted_label_height, bimodal = estimate_hueXvalue(image)
    else:
        predicted_label_height = predicted_label_Value
        bimodal = bimodal_value
    
    # get hot encode for color
    predicted_label_color = estimate_color(image)
    
    if print_stats:
        print('height: {}, bimodal: {}'.format(predicted_label_height, bimodal))
        print('color: {}'.format(predicted_label_color))
    
    #union of two hot_encoded labels
    union = matrix_multiplication(predicted_label_height, predicted_label_color)
    
    #if height label and color label match return encoder
    if predicted_label_height == predicted_label_color:
        predicted_label = predicted_label_height
        # check to make sure two colors arn't labeled
        if not double_encoded(predicted_label_height):
            return predicted_label
        
    # if height label and color label do not match
    #if double encoded
    if double_encoded(predicted_label_height):
        #use color if not double encoded
        if not double_encoded(predicted_label_color) and has_label(predicted_label_color):
            # check to make sure estimates do not conflict, and contain a union
            if np.sum(union) > 0:
                predicted_label = predicted_label_color
            else:
                #if double encoded on green or red choose red
                if predicted_label_height[0]*predicted_label_height[1] > 0:
                    predicted_label = [1, 0, 0]
            return predicted_label
        #if color is double encoded
        else:
            # if they are encoded the same choose red if not use the union
            if np.sum(union) == 1:
                predicted_label = union
            else:
                predicted_label = [1, 0, 0]
            return predicted_label
    # if height label is not double labelled and conflicts with color
    # choose color if available
    else:        
        if has_label(predicted_label_color) and predicted_label_color != [0, 1, 0]:
            predicted_label = predicted_label_color
        # if no color got off of value feature
        # if bimodal 
        else:
            if bimodal_value:
                predicted_label = predicted_label_height
            else:
                predicted_label = predicted_label_Value
    
    return predicted_label   

image, label, image_num = getImage(color='red', Random=True, showImg=False)
#image_num = random.randint(0, total_num_of_images-1)
#image_num = 757
test_im = STANDARDIZED_LIST[image_num][0]
test_label = STANDARDIZED_LIST[image_num][1]

encoder = estimate_label(test_im)
print('image: ', image_num)
print('guess: ', encoder)
print('actual: ', test_label)
plt.imshow(test_im)
plt.show()
