
class IdeaMetadataValues():

    def get_metadata_map(self):

        metadata_map = {}

        metadata_map['int'] = 1
        metadata_map['float'] = 2
        #metadata_map['<class \'double\'>'] = 3
        #metadata_map['<class \'char\'>'] = 4
        #metadata_map['<class \'short\'>'] = 5
        metadata_map['bool'] = 6
        metadata_map['str'] = 7

        metadata_map['list_int'] = 8
        #metadata_map['double_array'] = 9
        metadata_map['list_float'] = 10
        #metadata_map['short_array'] = 11
        #metadata_map['long_array'] = 12
        metadata_map['list_bool'] = 13
        metadata_map['list_str'] = 14

        return metadata_map
        