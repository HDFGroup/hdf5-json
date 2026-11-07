##############################################################################
# Copyright by The HDF Group.                                                #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of HSDS (HDF5 Scalable Data Service), Libraries and      #
# Utilities.  The full HSDS copyright notice, including                      #
# terms governing use, modification, and redistribution, is contained in     #
# the file COPYING, which can be found at the root of the source code        #
# distribution tree.  If you do not have access to this file, you may        #
# request a copy from help@hdfgroup.org.                                     #
##############################################################################
import unittest

from h5json.objid import isRootObjId, isValidUuid, validateUuid
from h5json.objid import createObjId, getCollectionForId, getUuidFromId, getHashTagForId
from h5json.objid import isObjId, isS3ObjKey, getS3Key, getObjId, isSchema2Id
from h5json.objid import getRootObjId, isValidChunkId, getClassForObjId, isCanonicalObjId


class IdUtilTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(IdUtilTest, self).__init__(*args, **kwargs)
        # main

    def testCreateObjId(self):
        id_len = 38  # 36 for uuid plus two for prefix ("g-", "d-")
        ids = set()  # we'll use this to verify we always get a unique id
        # create just a plain uuid...
        id = createObjId()
        self.assertEqual(len(id) + 2, id_len)
        # create a v2 root_id
        root_id = createObjId(obj_type="groups")
        self.assertEqual(len(root_id), id_len)
        for obj_type in ("groups", "datasets", "datatypes", "chunks"):
            for i in range(100):
                id = createObjId(obj_type=obj_type, root_id=root_id)
                self.assertEqual(len(id), id_len)
                self.assertTrue(id[0] in ("g", "d", "t", "c"))
                self.assertEqual(id[1], "-")
                ids.add(id)

        self.assertEqual(len(ids), 400)
        try:
            createObjId(obj_type="bad_class")
            self.assertTrue(False)  # should throw exception
        except ValueError:
            pass  # expected

    def testIsValidUuid(self):
        group1_id = "g-314d61b8-9954-11e6-a733-3c15c2da029e"      # orig schema
        group2_id = "g-314d61b8-995411e6-a733-3c15c2-da029e"
        root_id = "g-f9aaa28e-d42e10e5-7122-2a065c-a6986d"
        dataset1_id = "d-4c48f3ae-9954-11e6-a3cd-3c15c2da029e"    # orig schema
        dataset2_id = "d-4c48f3ae-995411e6-a3cd-3c15c2-da029e"
        ctype1_id = "t-8c785f1c-9953-11e6-9bc2-0242ac110005"      # orig schema
        ctype2_id = "t-8c785f1c-995311e6-9bc2-0242ac-110005"
        chunk1_id = "c-8c785f1c-9953-11e6-9bc2-0242ac110005_7_2"  # orig schema
        chunk2_id = "c-8c785f1c-995311e6-9bc2-0242ac-110005_7_2"
        domain_id = "mybucket/bob/mydata.h5"
        s3_domain_id = "s3://mybucket/bob/mydata.h5"
        file_domain_id = "file://mybucket/bob/mydata.h5"
        azure_domain_id = "https://myaccount.blob.core.windows.net/mybucket/bob/mydata.h5"
        valid_id_map = {
            group1_id: "a49be-g-314d61b8-9954-11e6-a733-3c15c2da029e",
            group2_id: "db/314d61b8-995411e6/g/a733-3c15c2-da029e/.group.json",
            dataset1_id: "26928-d-4c48f3ae-9954-11e6-a3cd-3c15c2da029e",
            dataset2_id: "db/4c48f3ae-995411e6/d/a3cd-3c15c2-da029e/.dataset.json",
            ctype1_id: "5a9cf-t-8c785f1c-9953-11e6-9bc2-0242ac110005",
            ctype2_id: "db/8c785f1c-995311e6/t/9bc2-0242ac-110005/.datatype.json",
            chunk1_id: "dc4ce-c-8c785f1c-9953-11e6-9bc2-0242ac110005_7_2",
            chunk2_id: "db/8c785f1c-995311e6/d/9bc2-0242ac-110005/7_2",
            domain_id: "bob/mydata.h5/.domain.json",
            s3_domain_id: "bob/mydata.h5/.domain.json",
            file_domain_id: "bob/mydata.h5/.domain.json",
            azure_domain_id: "bob/mydata.h5/.domain.json", }

        bad_ids = ("g-1e76d862", "/bob/mydata.h5")

        self.assertTrue(isValidUuid(group1_id))
        self.assertFalse(isSchema2Id(group1_id))
        self.assertTrue(isValidUuid(group1_id, obj_class="Group"))
        self.assertTrue(isValidUuid(group1_id, obj_class="group"))
        self.assertTrue(isValidUuid(group1_id, obj_class="groups"))
        self.assertTrue(isSchema2Id(root_id))
        self.assertTrue(isValidUuid(root_id, obj_class="Group"))
        self.assertTrue(isValidUuid(root_id, obj_class="group"))
        self.assertTrue(isValidUuid(root_id, obj_class="groups"))
        self.assertTrue(isRootObjId(root_id))
        self.assertTrue(isValidUuid(dataset1_id, obj_class="datasets"))
        self.assertFalse(isSchema2Id(dataset1_id))
        self.assertTrue(isValidUuid(ctype1_id, obj_class="datatypes"))
        self.assertFalse(isSchema2Id(ctype1_id))
        self.assertTrue(isValidUuid(chunk1_id, obj_class="chunks"))
        self.assertFalse(isSchema2Id(chunk1_id))
        self.assertTrue(isValidUuid(group2_id))
        self.assertTrue(isSchema2Id(group2_id))
        self.assertTrue(isValidUuid(group2_id, obj_class="Group"))
        self.assertTrue(isValidUuid(group2_id, obj_class="group"))
        self.assertTrue(isValidUuid(group2_id, obj_class="groups"))
        self.assertFalse(isRootObjId(group2_id))
        self.assertTrue(isValidUuid(dataset2_id, obj_class="datasets"))
        self.assertTrue(isSchema2Id(dataset2_id))
        self.assertTrue(isValidUuid(ctype2_id, obj_class="datatypes"))
        self.assertTrue(isSchema2Id(ctype2_id))
        self.assertTrue(isValidUuid(chunk2_id, obj_class="chunks"))
        self.assertTrue(isSchema2Id(chunk2_id))
        validateUuid(group1_id)
        try:
            isRootObjId(group1_id)
            self.assertTrue(False)
        except ValueError:
            # only works for v2 schema
            pass  # expected

        for item in valid_id_map:
            self.assertTrue(isObjId(item))
            s3key = getS3Key(item)
            self.assertTrue(s3key[0] != "/")
            self.assertTrue(isS3ObjKey(s3key))
            expected = valid_id_map[item]
            self.assertEqual(s3key, expected)
            if item.find("/") > 0:
                continue  # bucket name gets lost when domain ids get converted to s3keys
            objid = getObjId(s3key)
            self.assertEqual(objid, item)
        for item in bad_ids:
            self.assertFalse(isValidUuid(item))
            self.assertFalse(isObjId(item))

    def testGetCollection(self):
        group_id = "g-314d61b8-9954-11e6-a733-3c15c2da029e"
        dataset_id = "d-4c48f3ae-9954-11e6-a3cd-3c15c2da029e"
        ctype_id = "t-8c785f1c-9953-11e6-9bc2-0242ac110005"
        bad_id = "x-59647858-9954-11e6-95d2-3c15c2da029e"
        self.assertEqual(getCollectionForId(group_id), "groups")
        self.assertEqual(getCollectionForId(dataset_id), "datasets")
        self.assertEqual(getCollectionForId(ctype_id), "datatypes")
        self.assertEqual(getUuidFromId(group_id), "314d61b8-9954-11e6-a733-3c15c2da029e")
        try:
            getCollectionForId(bad_id)
            self.assertTrue(False)
        except ValueError:
            pass  # expected
        try:
            getCollectionForId(None)
            self.assertTrue(False)
        except ValueError:
            pass  # expected

    def testSchema2Id(self):
        root_id = createObjId("groups")
        group_id = createObjId("groups", root_id=root_id)
        dataset_id = createObjId("datasets", root_id=root_id)
        ctype_id = createObjId("datatypes", root_id=root_id)

        self.assertEqual(getCollectionForId(root_id), "groups")
        self.assertEqual(getCollectionForId(group_id), "groups")
        self.assertEqual(getCollectionForId(dataset_id), "datasets")
        self.assertEqual(getCollectionForId(ctype_id), "datatypes")
        chunk_id = "c" + dataset_id[1:] + "_1_2"
        chunk_partition_id = "c42-" + dataset_id[2:] + "_1_2"

        for id in (chunk_id, chunk_partition_id):
            try:
                getCollectionForId(id)
                self.assertTrue(False)
            except ValueError:
                pass  # expected
        valid_ids = (
            group_id,
            dataset_id,
            ctype_id,
            chunk_id,
            chunk_partition_id,
            root_id,
        )
        s3prefix = getS3Key(root_id)
        self.assertTrue(s3prefix.endswith("/.group.json"))
        s3prefix = s3prefix[: -(len(".group.json"))]
        for oid in valid_ids:
            self.assertTrue(len(oid) >= 38)
            parts = oid.split("-")
            self.assertEqual(len(parts), 6)
            self.assertTrue(oid[0] in ("g", "d", "t", "c"))
            self.assertTrue(isSchema2Id(oid))
            if oid == root_id:
                self.assertTrue(isRootObjId(oid))
            else:
                self.assertFalse(isRootObjId(oid))

            s3key = getS3Key(oid)
            self.assertTrue(s3key.startswith(s3prefix))
            self.assertEqual(getObjId(s3key), oid)
            self.assertTrue(isS3ObjKey(s3key))

    def testGetDataTypeId(self):
        test_uuid = "9b652223-83f8-11e5-b028-3c15c2da029e"
        test_ids = (
            "datatypes/9b652223-83f8-11e5-b028-3c15c2da029e",
            "datatypes/t-9b652223-83f8-11e5-b028-3c15c2da029e",
            "t-9b652223-83f8-11e5-b028-3c15c2da029e"
        )
        for test_id in test_ids:
            self.assertTrue(isValidUuid(test_id))
            self.assertEqual(getCollectionForId(test_id), "datatypes")
            self.assertEqual(getUuidFromId(test_id), test_uuid)
            self.assertEqual(getHashTagForId(test_id), "t-" + test_uuid)

    def testGetRootObjId(self):
        root_id = createObjId("groups")
        group_id = createObjId("groups", root_id=root_id)
        dataset_id = createObjId("datasets", root_id=root_id)
        ctype_id = createObjId("datatypes", root_id=root_id)
        chunk_id = "c" + dataset_id[1:] + "_1_2"

        # root id maps to itself
        self.assertEqual(getRootObjId(root_id), root_id)
        # objects created with root_id as their root should map back to root_id
        self.assertEqual(getRootObjId(group_id), root_id)
        self.assertEqual(getRootObjId(dataset_id), root_id)
        self.assertEqual(getRootObjId(ctype_id), root_id)
        self.assertEqual(getRootObjId(chunk_id), root_id)

        # only supported for v2 (schema 2) ids
        group1_id = "g-314d61b8-9954-11e6-a733-3c15c2da029e"  # orig (v1) schema
        try:
            getRootObjId(group1_id)
            self.assertTrue(False)
        except ValueError:
            pass  # expected

    def testIsValidChunkId(self):
        chunk1_id = "c-8c785f1c-9953-11e6-9bc2-0242ac110005_7_2"  # orig schema
        chunk2_id = "c-8c785f1c-995311e6-9bc2-0242ac-110005_7_2"  # v2 schema
        group_id = "g-314d61b8-9954-11e6-a733-3c15c2da029e"
        dataset_id = "d-4c48f3ae-9954-11e6-a3cd-3c15c2da029e"

        self.assertTrue(isValidChunkId(chunk1_id))
        self.assertTrue(isValidChunkId(chunk2_id))
        self.assertFalse(isValidChunkId(group_id))
        self.assertFalse(isValidChunkId(dataset_id))
        # not even a valid uuid
        self.assertFalse(isValidChunkId("not-a-valid-id"))

    def testGetClassForObjId(self):
        group_id = "g-314d61b8-9954-11e6-a733-3c15c2da029e"
        dataset_id = "d-4c48f3ae-9954-11e6-a3cd-3c15c2da029e"
        ctype_id = "t-8c785f1c-9953-11e6-9bc2-0242ac110005"
        chunk_id = "c-8c785f1c-9953-11e6-9bc2-0242ac110005_7_2"
        domain_id = "/bob/mydata.h5"

        self.assertEqual(getClassForObjId(domain_id), "domains")
        self.assertEqual(getClassForObjId(chunk_id), "chunks")
        self.assertEqual(getClassForObjId(group_id), "groups")
        self.assertEqual(getClassForObjId(dataset_id), "datasets")
        self.assertEqual(getClassForObjId(ctype_id), "datatypes")

        try:
            getClassForObjId(12345)
            self.assertTrue(False)
        except ValueError:
            pass  # expected

        try:
            getClassForObjId("")
            self.assertTrue(False)
        except ValueError:
            pass  # expected

    def testValidateUuidNonHex(self):
        root_id = "g-f9aaa28e-d42e10e5-7122-2a065c-a6986d"
        group_id = "g-314d61b8-995411e6-a733-3c15c2-da029e"
        upper_ids = (
            root_id[:20] + root_id[20:].upper(),
            group_id[:2] + group_id[2:].upper(),
            "314D61B8-9954-11E6-A733-3C15C2DA029E",
        )
        for obj_id in upper_ids:
            self.assertTrue(isValidUuid(obj_id))
            validateUuid(obj_id)
        self.assertTrue(isRootObjId(upper_ids[0]))

        fullwidth = str.maketrans("0123456789", "\uff10\uff11\uff12\uff13\uff14\uff15\uff16\uff17\uff18\uff19")
        arabic_indic = str.maketrans("0123456789", "\u0660\u0661\u0662\u0663\u0664\u0665\u0666\u0667\u0668\u0669")
        bad_ids = (
            "g-z14d61b8-995411e6-a733-3c15c2-da029e",
            "g-314d61b8-995411e6-a733-3c15c2-da029z",
            "g-314d61b8-9954-11e6-a733-3c15c2da0zzz",
            "zzzzzzzz-9954-11e6-a733-3c15c2da029e",
            "datasets/d-4c48f3ae-995411e6-a3cd-3c15c2-da0g9e",
            "c-8c785f1c-995311e6-9bc2-0242ac-1100x5_7_2",
            group_id[:20] + group_id[20:].translate(fullwidth),
            root_id[:20] + root_id[20:].translate(arabic_indic),
            group_id.translate(arabic_indic),
        )
        for obj_id in bad_ids:
            self.assertFalse(isValidUuid(obj_id), obj_id)
            with self.assertRaises(ValueError):
                validateUuid(obj_id)
        # the Unicode-digit alias of a root id is no longer a root id
        with self.assertRaises(ValueError):
            isRootObjId(root_id[:20] + root_id[20:].translate(arabic_indic))

    def testGetS3KeySlashForm(self):
        root_id = createObjId("groups")
        group_id = createObjId("groups", root_id=root_id)
        dataset_id = createObjId("datasets", root_id=root_id)
        ctype_id = createObjId("datatypes", root_id=root_id)
        slash_ids = (
            "groups/" + root_id,
            "groups/" + group_id,
            "groups/" + group_id[2:],
            "datasets/" + dataset_id,
            "datasets/" + dataset_id[2:],
            "datatypes/" + ctype_id,
            "datatypes/9b652223-83f8-11e5-b028-3c15c2da029e",
            "datatypes/t-9b652223-83f8-11e5-b028-3c15c2da029e",
        )
        for obj_id in slash_ids:
            self.assertTrue(isValidUuid(obj_id), obj_id)
            with self.assertRaises(ValueError):
                getS3Key(obj_id)
        # the unprefixed ids and domain paths still map to keys
        for obj_id in (root_id, group_id, dataset_id, ctype_id):
            self.assertEqual(getObjId(getS3Key(obj_id)), obj_id)
        self.assertEqual(getS3Key("mybucket/" + group_id), group_id + "/.domain.json")
        self.assertEqual(getS3Key("groups/bob/mydata.h5"), "bob/mydata.h5/.domain.json")
        self.assertEqual(getS3Key("s3://groups/" + group_id), group_id + "/.domain.json")

    def testIsCanonicalObjId(self):
        root_id = createObjId("groups")
        group_id = createObjId("groups", root_id=root_id)
        dataset_id = createObjId("datasets", root_id=root_id)
        ctype_id = createObjId("datatypes", root_id=root_id)
        chunk_id = "c" + dataset_id[1:] + "_1_2"
        for obj_id, obj_class in (
            (root_id, "groups"),
            (group_id, "Group"),
            (dataset_id, "datasets"),
            (ctype_id, "datatype"),
            (chunk_id, "chunks"),
            ("c" + dataset_id[1:] + "_0", "chunks"),
        ):
            self.assertTrue(isCanonicalObjId(obj_id), obj_id)
            self.assertTrue(isCanonicalObjId(obj_id, obj_class), obj_id)
            other = "datasets" if obj_class.lower() != "datasets" else "groups"
            self.assertFalse(isCanonicalObjId(obj_id, other), obj_id)
        self.assertFalse(isCanonicalObjId(group_id, "bad_class"))

        not_canonical = (
            None,
            42,
            "",
            createObjId(),  # bare uuid
            "g-314d61b8-9954-11e6-a733-3c15c2da029e",  # v1
            "a49be-g-314d61b8-9954-11e6-a733-3c15c2da029e",  # v1 hash tag
            "c-8c785f1c-9953-11e6-9bc2-0242ac110005_7_2",  # v1 chunk
            "groups/" + group_id,
            "groups/" + group_id[2:],
            "datasets/" + dataset_id,
            "datatypes/" + ctype_id,
            "datatypes/9b652223-83f8-11e5-b028-3c15c2da029e",
            group_id[:2] + group_id[2:].upper(),
            "G" + group_id[1:],
            group_id[:-1] + "z",
            group_id[:-1] + "\u0661",
            group_id + "\n",
            " " + group_id,
            group_id + "_1",  # chunk index on a group id
            "c" + dataset_id[1:],  # chunk id without an index
            "c" + dataset_id[1:] + "_",
            "c" + dataset_id[1:] + "_1_",
            "c" + dataset_id[1:] + "__1",
            "c" + dataset_id[1:] + "_-1",
            "c" + dataset_id[1:] + "_\u0661",
            "c42-" + dataset_id[2:] + "_1_2",  # partitioned chunk id
            "x" + group_id[1:],
        )
        for obj_id in not_canonical:
            self.assertFalse(isCanonicalObjId(obj_id), repr(obj_id))


if __name__ == "__main__":
    # setup test files

    unittest.main()
