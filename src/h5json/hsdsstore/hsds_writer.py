##############################################################################
# Copyright by The HDF Group.                                                #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of H5Serv (HDF5 REST Server) Service, Libraries and      #
# Utilities.  The full HDF5 REST Server copyright notice, including          #
# terms governing use, modification, and redistribution, is contained in     #
# the file COPYING, which can be found at the root of the source code        #
# distribution tree.  If you do not have access to this file, you may        #
# request a copy from help@hdfgroup.org.                                     #
##############################################################################
import logging
import time

from ..objid import getCollectionForId

from ..hdf5dtype import isVlen
from ..array_util import arrayToBytes, bytesArrayToList
from ..dset_util import getNumElements
from .. import selections
from ..h5writer import H5Writer
from .httpconn import HttpConn


class HSDSWriter(H5Writer):
    """
    This class can be used by HDF5DB to read content from an hdf5-json file
    """

    def __init__(
        self,
        domain_path,
        append=False,
        no_data=False,
        app_logger=None,
        endpoint=None,
        username=None,
        password=None,
        bucket=None,
        api_key=None,
        use_session=True,
        expire_time=0,
        max_objects=0,
        max_age=0,
        retries=3,
        timeout=30.0,
        track_order=False,
        owner=None,
        linked_domain=None

    ):
        if app_logger:
            self.log = app_logger
        else:
            self.log = logging.getLogger()

        if append:
            self._init = False
        else:
            self._init = True

        if no_data:
            self._no_data = True
        else:
            self._no_data = False

        self.log.debug("HSDSWriter init")

        kwargs = {}
        self.log.debug(f"    domain_path: {domain_path}")
        self.log.debug(f"    append: {append}")
        if endpoint:
            self.log.debug(f"    endpoint: {endpoint}")
            kwargs["endpoint"] = endpoint
        if username:
            self.log.debug(f"    username: {username}")
            kwargs["username"] = username
        if password:
            self.log.debug(f"    password: {'*' * len(password)}")
            kwargs["password"] = password
        if bucket:
            self.log.debug(f"    bucket: {bucket}")
            kwargs["bucket"] = bucket
        if api_key:
            self.log.debug(f"    apI_key: {'*' * len(api_key)}")
            kwargs["api_key"] = api_key
        if use_session:
            self.log.debug(f"    use_session: {use_session}")
            kwargs["user_session"] = use_session
        if expire_time:
            self.log.debug(f"    expire_time: {expire_time}")
            kwargs["expire_time"] = expire_time
        if max_objects:
            self.log.debug(f"    max_objects: {max_objects}")
            kwargs["max_objects"] = max_objects
        if max_age:
            self.log.debug(f"    max_age: {max_age}")
            kwargs["max_age"] = max_age
        if retries:
            self.log.debug(f"    retries: {retries}")
            kwargs["retries"] = retries
        if timeout:
            self.log.debug(f"    timeout: {timeout}")
            kwargs["timeout"] = timeout
        self._http_kwargs = kwargs  # save for when we create the connection

        super().__init__(domain_path, app_logger=app_logger)

        self._http_conn = None
        self._root_id = None
        self._append = append
        self._owner = owner
        self._track_order = track_order
        self._linked_domain = linked_domain
        self._domain_json = None
        self._last_flush_time = 0

    def open(self):
        """ setup domain for writing """
        if not self._db_ref:
            # no db set yet
            raise IOError("DB not set")

        if self._http_conn:
            http_conn = self._http_conn
        else:
            kwargs = self._http_kwargs
            kwargs["retries"] = 1  # tbd: test setting
            http_conn = HttpConn(self.filepath, **kwargs)
            if self._append:
                http_conn._mode = "a"
            self._http_conn = http_conn
            hsds_info = http_conn.serverInfo()
            self.log.debug(f"got hsds info: {hsds_info}")

        if not self._domain_json:
            # haven't fetched the domain json yet, do it now

            # try to do a GET from the domain
            req = "/"
            params = {}
            """
            if max_objects is None or max_objects > 0:
                # get object meta objects
                # TBD: have hsds support a max limit of objects to return
                params["getobjs"] = 1
                params["include_attrs"] = 1
                params["include_links"] = 1
            """

            domain_json = None
            rsp = http_conn.GET(req, params=params)

            if rsp.status_code not in (200, 404, 410):
                msg = f"Got status code: {rsp.status_code} on initial domain get"
                self.log.warning(msg)
                raise IOError(msg)

            if rsp.status_code == 200:
                if self._append:
                    # domain exists already
                    domain_json = rsp.json()
                    if "root" not in domain_json:
                        # this a folder not a domain
                        self.log.warning(f"folder: {self.filepath} has no root property")
                        http_conn.close()
                        raise IOError(404, "Location is a folder, not a file")
                else:
                    # not append - delete existing domain
                    self.log.info(f"sending delete request for {self.filepath}")
                    delete_rsp = http_conn.DELETE(req, params=params)
                    if delete_rsp.status_code not in (200, 410):
                        # failed to delete
                        http_conn.close()
                        raise IOError(rsp.status_code, rsp.reason)

            if not domain_json:
                # domain doesn't exist, create it
                body = {}
                if self.db.root_id:
                    # initialize domain using the db's root_id
                    body["root_id"] = self.db.root_id
                if self._owner:
                    body["owner"] = self._owner
                if self._linked_domain:
                    body["linked_domain"] = self._linked_domain
                if self._track_order:
                    create_props = {"CreateOrder": 1}
                    group_body = {"creationProperties": create_props}
                    body["group"] = group_body
                rsp = http_conn.PUT(req, params=params, body=body)
                if rsp.status_code != 201:
                    http_conn.close()
                    raise IOError(rsp.status_code, rsp.reason)
                domain_json = rsp.json()
                self.log.info(f"got rsp on PUT domain: {domain_json}")
                if "root" not in domain_json:
                    http_conn.close()
                    raise IOError(404, "Unexpected error")

            self.log.debug(f"got domain_json: {domain_json}")

            if "root" not in domain_json:
                http_conn.close()
                raise IOError(404, "Location is a folder, not a file")

            root_id = domain_json["root"]

            self._root_id = root_id

            if "limits" in domain_json:
                self._limits = domain_json["limits"]
            else:
                self._limits = None
            if "version" in domain_json:
                self._version = domain_json["version"]
            else:
                self._version = None

            self._domain_json = domain_json

        return self._root_id

    @property
    def http_conn(self):
        return self._http_conn

    def getDatasetSize(self, dset_id):
        """ Return the size of the given dataset """

        dset_json = self.db.getObjectById(dset_id)
        num_elements = getNumElements(dset_json)
        dtype = self.db.getDtype(dset_json)
        if isVlen(dtype):
            item_size = 1024  # random guess at size of variable length types
        else:
            item_size = dtype.itemsize
        return num_elements * item_size

    def createObjects(self, obj_ids):
        """ create the objects referenced in obj_ids """

        MAX_INIT_SIZE = 4096  # max size to include init values in dataset creation

        def multiPost(items):
            self.log.debug(f"hsds_writer> POST request {collection} for {len(items)} objects")
            for item in items:
                self.log.debug(f"hsds_writer> POST item: {item}")
            post_rsp = self.http_conn.POST("/" + collection, items)
            self.log.debug(f"hsds_writer> POST post_rsp.status_code: {post_rsp.status_code}")
            items.clear()

        self.log.debug(f"hsds_writer> createObjects, {len(obj_ids)} objects")
        MAX_OBJECTS_PER_REQUEST = 3
        collections = ("groups", "datasets", "datatypes")
        col_items = {}
        dset_value_update_ids = set()
        for collection in collections:
            col_items[collection] = []

        for obj_id in obj_ids:
            if obj_id == self._root_id:
                continue  # this was created when the domain was
            collection = getCollectionForId(obj_id)
            obj_json = self.db.getObjectById(obj_id)
            item = {"id": obj_id}
            self.log.debug(f"create id: {obj_id}")
            for key in obj_json:  # ("links", "attributes"):
                if key == "updates":
                    # not part of the obj json
                    continue
                if key == "attributes":
                    # will update attribute later
                    continue
                if key == "links":
                    # links will also be updated later
                    continue
                if key == "shape":
                    # just send the dims, not the shape json
                    shape_json = obj_json["shape"]
                    if shape_json["class"] == "H5S_SIMPLE":
                        dims = shape_json["dims"]
                        item[key] = dims
                else:
                    # just copy the key value directly
                    item[key] = obj_json[key]

            # initialize dataset values if provided and not too large
            if "updates" in obj_json:
                updates = obj_json["updates"]
                if updates and len(updates) == 1 and self.getDatasetSize(obj_id) < MAX_INIT_SIZE:
                    sel, arr = updates[0]
                    if sel.select_type == selections.H5S_SELECT_ALL:
                        value = bytesArrayToList(arr)
                        item["value"] = value
                        updates.clear()  # reset the update list
                if updates:
                    dset_value_update_ids.add(obj_id)  # will set dataset value below

            # add to the list of new items for the given collection
            items = col_items[collection]
            items.append(item)

            if len(items) == MAX_OBJECTS_PER_REQUEST:
                multiPost(items)

        # handle any remainder items
        for collection in collections:
            items = col_items[collection]
            if items:
                multiPost(items)

        # write any initial dataset values
        if dset_value_update_ids:
            self.updateValues(dset_value_update_ids)

    def deleteObjects(self, obj_ids):
        """ remove the given obj ids from the HSDS store """

        # no multi-delete operation yet, so delete one by one
        for obj_id in obj_ids:
            collection = getCollectionForId(obj_id)
            req = f"/{collection}/{obj_id}"
            http_rsp = self.http_conn.DELETE(req)
            if http_rsp.status_code not in (200, 410):
                self.log.error(f"got {http_rsp.status_code} for DELETE {req}")

    def updateLinks(self, grp_ids):
        """ update any modified links of the given objects """

        self.log.debug("hsds_writer> updateLinks")
        items = {}  # dict which will hold a map of grp ids to links to create
        count = 0

        for grp_id in grp_ids:
            if getCollectionForId(grp_id) != "groups":
                continue  # ignore datasets and datatypes
            grp_json = self.db.getObjectById(grp_id)
            grp_links = grp_json["links"]
            for link_title in grp_links:
                link_json = grp_links[link_title]
                if "created" not in link_json:
                    self.log.error(f"hsds_writer> expected created timestamp in link: {link_json}")
                created = link_json["created"]
                if created > self._last_flush_time:
                    self.log.debug(f"hsds_writer> {grp_id}: new link: {link_title}")
                    count += 1
                    # new link, add to our list
                    if grp_id not in items:
                        items[grp_id] = {"links": {}}
                    links = items[grp_id]["links"]
                    link_class = link_json["class"]
                    new_link = {"class": link_class}
                    # convert to hsds representation
                    if link_class == "H5L_TYPE_HARD":
                        new_link["id"] = link_json["id"]
                    elif link_class == "H5L_TYPE_SOFT":
                        new_link["h5path"] = link_json["h5path"]
                    elif link_class == "H5L_TYPE_EXTERNAL":
                        new_link["h5path"] = link_json["h5path"]
                        new_link["h5domain"] = link_json["file"]  # use h5domain for file key
                    elif link_class == "H5L_TYPE_USER_DEFINED":
                        self.log.warning(f"ignoring user-defined link: {link_title}")
                        continue
                    else:
                        raise IOError(f"unexpected link class: {link_class}")
                    links[link_title] = new_link
                    self.log.debug(f"setting link {link_title} to {new_link}")

        if items:
            body = {"grp_ids": items}
            put_rsp = self.http_conn.PUT("/groups/" + self._root_id + "/links", body=body)
            if put_rsp.status_code not in (200, 201):
                self.log.error(f"failed to update links for request: {body}")
                raise IOError("hsds_writer unable to update links")
            else:
                self.log.debug(f"hsds_writer> {grp_id} {count} links updated")

    def updateAttributes(self, obj_ids):
        """ update any modified links of the given objects """

        self.log.debug("hsds_writer> updateAttributes")
        items = {}  # dict which will hold a map of objects ids to attributes to create
        count = 0

        for obj_id in obj_ids:
            obj_json = self.db.getObjectById(obj_id)
            obj_attrs = obj_json["attributes"]
            for attr_name in obj_attrs:
                attr_json = obj_attrs[attr_name]
                if "created" not in attr_json:
                    self.log.error(f"hsds_writer> expected created timestamp in attr: {attr_json}")
                created = attr_json["created"]
                if created > self._last_flush_time:
                    self.log.debug(f"hsds_writer> {obj_id} attribute {attr_name} created")
                    count += 1
                    # new attribute, add to our list
                    if obj_id not in items:
                        items[obj_id] = {"attributes": {}}
                    attrs = items[obj_id]["attributes"]
                    attrs[attr_name] = attr_json

        if items:
            body = {"obj_ids": items}
            req = f"/groups/{self._root_id}/attributes"
            put_rsp = self.http_conn.PUT(req, body=body)
            if put_rsp.status_code not in (200, 201):
                self.log.error(f"hsds_writer> put {req} failed, status: {put_rsp.status_code}")
            else:
                self.log.debug(f"hsds_writer> {count} attributes updated")

    def updateValue(self, dset_id, sel, arr):
        """ update the given dataset using selection and array """
        self.log.debug("hsds_writer> updateValue")
        params = {}
        data = arrayToBytes(arr)
        self.log.debug(f"writing binary data, {len(data)} bytes")

        if sel.select_type != selections.H5S_SELECT_ALL:
            select_param = sel.getQueryParam()
            self.log.debug(f"got select query param: {select_param}")
            params["select"] = select_param

        req = f"/datasets/{dset_id}/value"
        rsp = self.http_conn.PUT(req, body=data, params=params, format="binary")
        if rsp.status_code != 200:
            self.log.error(f"PUT {req} returned error: {rsp.status_code}")
        else:
            self.log.debug(f"PUT {len(data)} bytes successful")

    def updateValues(self, dset_ids):
        """ write any pending dataset values """

        self.log.debug("hsds_writer> updateValues")
        for dset_id in dset_ids:
            if getCollectionForId(dset_id) != "datasets":
                continue  # ignore groups and datatypes
            dset_json = self.db.getObjectById(dset_id)
            if "updates" not in dset_json:
                continue
            updates = dset_json["updates"]
            if updates:
                self.log.debug(f"hsds_writer> {dset_id} update count: {len(updates)}")
                for (sel, arr) in updates:
                    self.updateValue(dset_id, sel, arr)
                updates.clear()


    def flush(self):
        """ Write dirty items """
        if self.closed:
            # no db set yet
            self.log.warning("hsds_writer> flush called but no db")
            return IOError("writer is closed")
        if not self._http_conn:
            self.log.warning("hsds_writer no http connection")
            raise IOError("no http connection")
        
        self.log.info("hsds_writer.flush()")
        self.log.debug(f"    new object count: {len(self.db.new_objects)}")
        self.log.debug(f"    dirty object count: {len(self.db.dirty_objects)}")
        self.log.debug(f"    deleted object count: {len(self.db.deleted_objects)}")
        root_id = self._root_id
        dirty_ids = self.db.dirty_objects.copy()
        if self._init:
            # initialize objects
            self.log.debug(f"hsds_writer> flush -- init is True self.db: {len(self.db.db)} objects")
            self.db.readAll()
            self.log.debug(f"hsds_writer>flush, init after readAll, {len(self.db.db)} objects")
            obj_ids = set(self.db.db.keys())
            obj_ids.remove(root_id)  # root group created when domain was
            self.log.debug(f"init createObjects: {obj_ids}")
            self.createObjects(obj_ids)
            dirty_ids.update(obj_ids)
            dirty_ids.add(root_id)  # add back root for attribute and link creation
            self._init = False
        elif self.db.new_objects:
            self.log.debug(f"hsds_writer> {len(self.db.new_objects)} objects to create")
            for obj_id in self.db.new_objects:
                self.log.debug(f"hsds_writer> new obj id: {obj_id}")
            self.createObjects(self.db.new_objects)
            dirty_ids.update(self.db.new_objects)
        else:
            self.log.debug("no new objects to persist")

        if dirty_ids:
            self.log.debug(f"hsds_writer> dirty ids: {dirty_ids}")
            self.updateLinks(dirty_ids)
            self.updateAttributes(dirty_ids)
            if not self._no_data:
                self.updateValues(dirty_ids)

        if self.db.deleted_objects:
            self.log.debug(f"deleted ids: {self.db.deleted_objects}")
            self.deleteObjects(self.db.deleted_objects)
        
        self._last_flush_time = time.time()
        self.log.debug("hsds_writer> flush successful")
        # all objects written successfully
        return True

    def close(self):
        # over-ride of H5Writer method
        self.flush()

    def isClosed(self):
        """ return closed status """
        return False if self._http_conn else True

    def get_root_id(self):
        """ Return root id """
        return self._root_id
