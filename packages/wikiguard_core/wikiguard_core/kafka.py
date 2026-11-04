"""Publishing to Redpanda, which speaks the Kafka API.

Idempotent producing stops a network retry from writing a message twice. A restart can still
re-send an edit, so consumers also treat rev_id as the key and upsert.
"""

import time
from typing import Protocol

from confluent_kafka import KafkaError, KafkaException, Message, Producer
from confluent_kafka.admin import AdminClient, NewTopic  # type: ignore[attr-defined]

EDITS_RAW = "edits.raw"
EDITS_DLQ = "edits.dlq"
EDITS_SCORED = "edits.scored"
LABELS_FINAL = "labels.final"
ONE_GIB = 1024**3
# Size limits come first because disk is the scarce thing on one VM. Time is only a backstop.
TOPIC_RETENTION_BYTES = {
    EDITS_RAW: ONE_GIB,
    EDITS_DLQ: ONE_GIB // 4,
    EDITS_SCORED: ONE_GIB // 2,
    LABELS_FINAL: ONE_GIB // 2,
}
TOPIC_RETENTION_MS = 7 * 24 * 60 * 60 * 1000
BUFFER_FULL_WAIT_SECONDS = 30.0
MAX_BUFFERED_MESSAGES = 10_000


class PublishError(RuntimeError):
    """Redpanda didn't confirm a message, so the caller must not record it as sent."""


class Publisher(Protocol):
    def publish(self, topic: str, key: str, value: bytes) -> None: ...

    def flush(self, timeout: float = 30.0) -> None: ...


class KafkaPublisher:
    def __init__(self, bootstrap_servers: str) -> None:
        self.producer = Producer(
            {
                "bootstrap.servers": bootstrap_servers,
                "enable.idempotence": True,
                "acks": "all",
                "linger.ms": 50,
                "compression.type": "zstd",
                # A bounded buffer: if Redpanda is down, stop reading rather than fill memory.
                "queue.buffering.max.messages": MAX_BUFFERED_MESSAGES,
                "message.timeout.ms": 120_000,
            }
        )
        self.failures: list[str] = []

    def _on_delivery(self, error: KafkaError | None, _message: Message) -> None:
        if error is not None:
            self.failures.append(error.name())

    def publish(self, topic: str, key: str, value: bytes) -> None:
        deadline = time.monotonic() + BUFFER_FULL_WAIT_SECONDS
        while True:
            try:
                self.producer.produce(topic, key=key, value=value, on_delivery=self._on_delivery)
                self.producer.poll(0)
                return
            except BufferError as error:
                if time.monotonic() > deadline:
                    raise PublishError("the send buffer stayed full") from error
                self.producer.poll(1)

    def flush(self, timeout: float = 30.0) -> None:
        remaining = self.producer.flush(timeout)
        if remaining:
            raise PublishError(f"{remaining} messages still unconfirmed")
        if self.failures:
            failures, self.failures = self.failures, []
            raise PublishError(f"delivery failed: {sorted(set(failures))}")


def ensure_topics(bootstrap_servers: str) -> None:
    """Create topics with their limits. An existing topic is left as it is."""
    admin = AdminClient({"bootstrap.servers": bootstrap_servers})
    topics = [
        NewTopic(
            name,
            num_partitions=1,
            replication_factor=1,
            config={"retention.bytes": str(size), "retention.ms": str(TOPIC_RETENTION_MS)},
        )
        for name, size in TOPIC_RETENTION_BYTES.items()
    ]
    for future in admin.create_topics(topics).values():
        try:
            future.result()
        except KafkaException as error:
            if error.args[0].code() != KafkaError.TOPIC_ALREADY_EXISTS:
                raise
