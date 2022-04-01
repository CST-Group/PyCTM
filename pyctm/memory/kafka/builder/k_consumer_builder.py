from confluent_kafka import Consumer

from pyctm.memory.kafka.topic_config_provider import TopicConfigProvider


class KConsumerBuilder:

    @staticmethod
    def build_consumer(broker, consumer_group_id, topic):
        consumer = Consumer({
            'bootstrap.servers': broker,
            'group.id': consumer_group_id,
            'auto.offset.reset': 'earliest'
        })

        consumer.subscribe([topic])

        return consumer

    @staticmethod
    def generate_consumers(topic_configs, consumer_group_id):

        consumers = {}

        for topic_config in topic_configs:

            if topic_config.regex_pattern is not None:
                print('Regex pattern %s identified.' %
                      topic_config.regex_pattern)

                if not topic_config.regex_pattern:

                    found_topic_configs = TopicConfigProvider.generate_topic_configs_regex_pattern(
                        topic_config.broker, topic_config.regex_pattern, topic_config.class_name)

                    if len(found_topic_configs) == 0:
                        raise Exception(
                            'Topic regex not found - pattern - %s' % topic_config.regex_pattern)

                    regex_pattern_consumers = KConsumerBuilder.generate_consumers(
                        found_topic_configs, consumer_group_id)

                    for key, value in regex_pattern_consumers.items():
                        consumers[key] = value

                return

            print('Creating consumer for topic configuration - Name: %s - Broker: %s - Class: %s - Behavior Type: %s'
                  % (topic_config.name, topic_config.broker, topic_config.class_name,
                     topic_config.k_distributed_memory_behavior))

            consumer = KConsumerBuilder.build_consumer(
                topic_config.broker, consumer_group_id, topic_config)

            print('Consumer created for topic %s.' % topic_config.name)

            consumers[topic_config] = consumer

        return consumers
