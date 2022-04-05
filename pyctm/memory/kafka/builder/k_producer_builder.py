from confluent_kafka import Producer
from confluent_kafka.admin import AdminClient
from confluent_kafka.cimpl import NewTopic


class KProducerBuilder:

    @staticmethod
    def build_producer(broker):
        producer = Producer({
            'bootstrap.servers': broker
        })
        return producer

    @staticmethod
    def check_topic_exist(broker, topic):
        kafka_admin = AdminClient({"bootstrap.servers": broker})
        topic_metadata = kafka_admin.list_topics()
        if topic_metadata.topics.get(topic) is None:
            new_kafka_topic = NewTopic(topic, num_partitions=1, replication_factor=1)
            kafka_admin.create_topics([new_kafka_topic])

    @staticmethod
    def generate_producers(topic_configs):
        producers = []

        for topic_config in topic_configs:
            print('Creating producer for topic configuration - Name: %s - Broker: %s - Class: %s - Behavior Type: %s' %
                  (topic_config.name,
                   topic_config.broker,
                   topic_config.class_name,
                   topic_config.k_distributed_memory_behavior.name))

            producer = KProducerBuilder.build_producer(topic_config.broker)

            print('Producer created fo topic %s.' % topic_config.name)

            producers.append(producer)

        return producers
