from shape_detect import calculate

def determine_face_shape(forehead_length, face_width, jawline_length, face_length):
    length_to_width_ratio = face_length / face_width

    if length_to_width_ratio > 1.5:
        if abs(jawline_length - forehead_length) < 0.1 * jawline_length:
            return "Rectangular"  
        return "Oblong" 

    if length_to_width_ratio > 1.3:
        if forehead_length > jawline_length:
            return "Oval"
        else:
            return "Heart"  

    if abs(face_length - face_width) <= 0.1 * face_length:
        return "Round"  

    if length_to_width_ratio < 1.2:
        if abs(jawline_length - forehead_length) < 0.1 * jawline_length:
            return "Square"  

    return "Diamond" 

