from collections.abc import Iterable

class ArrayValueValidation:

    @staticmethod
    def is_array(clazz):
        return "list" in clazz;

    @staticmethod
    def is_list(clazz):
        return issubclass(clazz, list)

    @staticmethod
    def is_string_array(clazz):
        return issubclass(clazz, list) and all(issubclass(subclazz, str) for subclazz in clazz.__args__)

    @staticmethod
    def is_primitive(clazz):
        return clazz in ["int", "float", "bool", "bytes"]

    @staticmethod
    def is_string(object):
        return isinstance(object, str)

    @staticmethod
    def is_string(clazz):
        return clazz == "str"

    def compare_value(self, new_value, value):
        if len(new_value) == len(value):
            return all(nv == v for nv, v in zip(new_value, value))
        return False