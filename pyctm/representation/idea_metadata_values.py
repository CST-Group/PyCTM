
class IdeaMetadataValues():

    def get_metadata_map(self):

        metadata_map = {}

        metadata_map['int'] = 1
        metadata_map['float'] = 3
        metadata_map['double'] = 3
        metadata_map['char'] = 4
        metadata_map['short'] = 5
        metadata_map['bool'] = 6
        metadata_map['str'] = 7       

        metadata_map['list_int'] = 8
        metadata_map['list_double'] = 9
        metadata_map['list_float'] = 9
        metadata_map['list_short'] = 11
        metadata_map['list_long'] = 12        
        metadata_map['list_bool'] = 13
        metadata_map['list_str'] = 14

        return metadata_map
    
    def is_array(metadata):
        return True if metadata > 7 else False
    
    def is_primitive(metadata):
        return True if metadata < 7 else False

    def is_bool(metadata):
        return True if metadata == 6 else False

    def is_string(metadata):
        return True if metadata == 7 or metadata == 4 else False
    
    def is_string_array(metadata):
        return True if metadata == 14 else False
    
    def is_bool_array(metadata):
        return True if metadata == 13 else False

    def get_metadata_key(self, value):

        metadata = self.get_metadata_map()

        for index in range(len(metadata.values())):
            if metadata.values()[index] == value:
                return metadata.keys()[index]
        
        return None
    
    
        