import tempfile
import unittest
from pathlib import Path

from eva_agentic.resources import ResourceAllocator


class ResourceAllocatorTest(unittest.TestCase):
    def test_allocates_distinct_gpu_slots(self) -> None:
        allocator = ResourceAllocator({"gpu": [0, 1]}, Path("/tmp/eva-locks"))

        with tempfile.TemporaryDirectory() as directory:
            allocator = ResourceAllocator({"gpu": [0, 1]}, Path(directory))
            first = allocator.allocate({"gpu": 1})
            second = allocator.allocate({"gpu": 1})

            self.assertEqual(first.grants[0].slot, 0)
            self.assertEqual(second.grants[0].slot, 1)
            first.release()
            second.release()

    def test_exhausted_resource_blocks_allocation_until_released(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            allocator = ResourceAllocator({"gpu": [0]}, Path(directory))
            lease = allocator.allocate({"gpu": 1})

            with self.assertRaisesRegex(RuntimeError, "exhausted: gpu"):
                allocator.allocate({"gpu": 1})

            lease.release()
            retry = allocator.allocate({"gpu": 1})
            self.assertEqual(retry.grants[0].slot, 0)
            retry.release()

    def test_allocates_multiple_kinds_independently(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            allocator = ResourceAllocator(
                {"gpu": [0, 1], "port": [8000, 8001]}, Path(directory)
            )

            with allocator.allocate({"gpu": 1, "port": 2}) as lease:
                self.assertEqual(
                    [(grant.kind, grant.slot) for grant in lease.grants],
                    [("gpu", 0), ("port", 8000), ("port", 8001)],
                )

            retry = allocator.allocate({"gpu": 2, "port": 1})
            retry.release()

    def test_rejects_unknown_or_invalid_requirements(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            allocator = ResourceAllocator({"gpu": [0]}, Path(directory))

            with self.assertRaisesRegex(ValueError, "unknown resource kinds"):
                allocator.allocate({"memory": 1})
            with self.assertRaisesRegex(ValueError, "must be a positive"):
                allocator.allocate({"gpu": 0})
