"""Against real Redpanda. Skipped unless TEST_KAFKA_BOOTSTRAP is set (127.0.0.1:19092 locally)."""

import os
import time
import uuid

import pytest
from confluent_kafka import Consumer
from confluent_kafka.admin import AdminClient, ConfigResource
from wikiguard_core.kafka import EDITS_RAW, KafkaPublisher, ensure_topics

pytestmark = pytest.mark.integration

BOOTSTRAP = os.environ.get("TEST_KAFKA_BOOTSTRAP", "")
skip_without_redpanda = pytest.mark.skipif(not BOOTSTRAP, reason="TEST_KAFKA_BOOTSTRAP is not set")


@skip_without_redpanda
def test_topics_get_size_based_retention() -> None:
    ensure_topics(BOOTSTRAP)
    ensure_topics(BOOTSTRAP)  # Creating twice must be harmless.

    admin = AdminClient({"bootstrap.servers": BOOTSTRAP})
    resource = ConfigResource(ConfigResource.Type.TOPIC, EDITS_RAW)
    config = admin.describe_configs([resource])[resource].result()

    assert int(config["retention.bytes"].value) == 1024**3


@skip_without_redpanda
def test_published_message_arrives_with_its_key() -> None:
    ensure_topics(BOOTSTRAP)
    marker = uuid.uuid4().hex
    publisher = KafkaPublisher(BOOTSTRAP)

    publisher.publish(EDITS_RAW, marker, f'{{"marker": "{marker}"}}'.encode())
    publisher.flush()

    consumer = Consumer(
        {
            "bootstrap.servers": BOOTSTRAP,
            "group.id": f"test-{marker}",
            "auto.offset.reset": "earliest",
        }
    )
    consumer.subscribe([EDITS_RAW])
    try:
        # A time limit rather than a count, since the topic can hold many earlier messages.
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            message = consumer.poll(1.0)
            if message is not None and not message.error() and message.key() == marker.encode():
                assert marker.encode() in message.value()
                return
    finally:
        consumer.close()
    pytest.fail("the message never arrived")
