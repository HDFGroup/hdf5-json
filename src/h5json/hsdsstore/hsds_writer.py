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

from ..objid import getCollectionForId, getUuidFromId

from ..hdf5dtype import createDataType
from ..array_util import jsonToArray, bytesToArray
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

    def open(self):
        """ setup domain for writing """

        if self._http_conn:
            http_conn = self._http_conn
        else:
            kwargs = self._http_kwargs
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
                    body["linked_domain"] = linked_domain
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
    
    def flush(self):
        """ Write dirty items """

        if not self.db:
            # no db set yet
            return False
        self.log.info("hsds_writer.flush()")
        self.log.debug(f"    new object count: {len(self.db.new_objects)}")
        self.log.debug(f"    dirty object count: {len(self.db.dirty_objects)}")
        self.log.debug(f"    deleted object count: {len(self.db.deleted_objects)}")

        #root_id = self.db.root_id
        if self._init:
            # initialize all existing objects
            self.log.debug("flush -- init is true")
            for obj_id in self.db:
                self.log.debug(f"init: {obj_id}")
            self._init = False
        elif self.db.new_objects:
            for obj_id in self.db.new_objects:
                self.log.debug(f"new obj id: {obj_id}")

        for obj_id in self.db.dirty_objects:
            self.log.debug(f"dirty object id: {obj_id}")

        for obj_id in self.db.deleted_objects:
            self.log.debug(f"deleted object: {obj_id}")
        
        return True  # all objects written successfully

    def close(self):
        # over-ride of H5Writer method
        self.flush()
        self.http_conn.close()
        self._http_conn = None

    def isClosed(self):
        """ return closed status """
        return False if self._http_conn else True

    def get_root_id(self):
        """ Return root id """
        return self._root_id
