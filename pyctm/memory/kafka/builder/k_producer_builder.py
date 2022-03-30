from confluent_kafka import Producer
from memory.kafka.topic_config_provider import TopicConfigProvider


class KProducerBuilder():

    @staticmethod
    def build_producer(broker):
        producer = Producer({'bootstrap.servers': broker})
        return producer

    @staticmethod
    def generate_producers(topic_configs):

        producers = []

        for topic_config in topic_configs:
            print('Creating producer for topic configuration - Name: %s - Broker: %s - Class: %s - Behavior Type: %s',
                  (topic_config.name,
                   topic_config.broker,
                   topic_config.class_name,
                   topic_config.k_distributed_memory_behavior.name))

            producer = KProducerBuilder.build_producer(topic_config.broker)

            print('Producer created fo topic %s.' % topic_config.name)

            producers.append(producer)

        return producers
