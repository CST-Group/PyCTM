import re
from confluent_kafka.admin import AdminClient
from pyctm.memory.kafka.config.topic_config import TopicConfig
from pyctm.memory.kafka.k_distributed_memory_behavior import KDistributedMemoryBehavior


class TopicConfigProvider:

    @staticmethod
    def generate_topic_configs_regex_pattern(broker, regex_pattern, class_name):

        k_admin = AdminClient({
            'bootstrap.servers': broker
        })

        cluster_metadata = k_admin.list_groups().topics

        if len(cluster_metadata.items()) == 0:
            raise Exception('Topics not found. Review regex pattern.')

        found_topics = []

        for key, _ in cluster_metadata.items():
            if re.search(regex_pattern, key):
                found_topics.append(TopicConfig(
                    key, broker, KDistributedMemoryBehavior.PULLED, regex_pattern, class_name))

        return found_topics
