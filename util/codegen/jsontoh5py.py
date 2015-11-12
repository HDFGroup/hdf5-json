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
import json
import argparse
from sets import Set
import posixpath as pp
import six

if six.PY3:
    unicode = str

"""
jsontoh5py - output Python code that generates HDF5 file based on given
JSON input
"""

# Global variables...
variable_names = Set()
dimensions = list()
dimscales = dict()


def getNumpyTypename(hdf5TypeName, typeClass=None):
    predefined_int_types = {
          'H5T_STD_I8':  'i1',
          'H5T_STD_U8':  'u1',
          'H5T_STD_I16': 'i2',
          'H5T_STD_U16': 'u2',
          'H5T_STD_I32': 'i4',
          'H5T_STD_U32': 'u4',
          'H5T_STD_I64': 'i8',
          'H5T_STD_U64': 'u8'
    }
    predefined_float_types = {
          'H5T_IEEE_F32': 'f4',
          'H5T_IEEE_F64': 'f8'
    }
    if len(hdf5TypeName) < 3:
        raise Exception("Type Error: invalid type: %s" % hdf5TypeName)
    endian = '<'  # default endian
    key = hdf5TypeName
    if hdf5TypeName.endswith('LE'):
        key = hdf5TypeName[:-2]
    elif hdf5TypeName.endswith('BE'):
        key = hdf5TypeName[:-2]
        endian = '>'

    if key in predefined_int_types and (typeClass is None or
                                        typeClass == 'H5T_INTEGER'):
        return endian + predefined_int_types[key]
    if key in predefined_float_types and (typeClass is None or
                                          typeClass == 'H5T_FLOAT'):
        return endian + predefined_float_types[key]
    raise TypeError("Type Error: invalid type")


def getBaseDataType(typeItem):
    code = "dt = "
    if type(typeItem) == str or type(typeItem) == unicode:
        # should be one of the predefined types
        dtName = getNumpyTypename(typeItem)
        code += "np.dtype('" + dtName + "')"
        return code

    if type(typeItem) != dict:
        raise TypeError("Type Error: invalid type")

    code += 'np.dtype({})'.format(_dtype(typeItem))

    return code


def _dtype(typeItem):
    """Helper function for generating numpy.dtype() code.

    typeItem is a dict with HDF5/JSON datatype description.
    """
    typeClass = typeItem['class']
    shape = ''
    if 'dims' in typeItem:
        shp_key = 'dims'
    else:
        shp_key = 'shape'
    if shp_key in typeItem:
        dims = None
        if type(typeItem[shp_key]) == int:
            dims = (typeItem[shp_key],)  # make into a tuple
        elif type(typeItem[shp_key]) not in (list, tuple):
            raise TypeError("expected list or integer for %s" % shp_key)
        else:
            dims = typeItem[shp_key]
        shape = str(tuple(dims))

    code = ''
    if typeClass == 'H5T_INTEGER':
        if 'base' not in typeItem:
            raise KeyError("'base' not provided")
        baseType = getNumpyTypename(typeItem['base'], typeClass='H5T_INTEGER')
        code += "'{}{}'".format(shape, baseType)
    elif typeClass == 'H5T_FLOAT':
        if 'base' not in typeItem:
            raise KeyError("'base' not provided")
        baseType = getNumpyTypename(typeItem['base'], typeClass='H5T_FLOAT')
        code += "'{}{}'".format(shape, baseType)
    elif typeClass == 'H5T_STRING':
        if 'length' not in typeItem:
            raise KeyError("'length' not provided")
        if 'charSet' not in typeItem:
            raise KeyError("'charSet' not provided")

        if typeItem['length'] == 'H5T_VARIABLE':
            if shape:
                raise TypeError(
                    "ArrayType is not supported for variable len types")
            if typeItem['charSet'] == 'H5T_CSET_ASCII':
                code += "h5py.special_dtype(vlen=str)"
            elif typeItem['charSet'] == 'H5T_CSET_UTF8':
                code += "h5py.special_dtype(vlen=unicode)"
            else:
                raise TypeError("unexpected 'charSet' value")
        else:
            # fixed size ascii string
            nStrSize = typeItem['length']
            if type(nStrSize) != int:
                raise TypeError("expecting integer value for 'length'")
            code += "'{}S{}'".format(shape, nStrSize)
    elif typeClass == 'H5T_VLEN':
        if shape:
            raise TypeError(
                "ArrayType is not supported for variable len types")
        if 'base' not in typeItem:
            raise KeyError("'base' not provided")
        vlenBaseType = typeItem['base']
        baseType = getNumpyTypename(vlenBaseType['base'],
                                    typeClass=vlenBaseType['class'])
        code += "h5py.special_dtype(vlen=np.dtype('" + baseType + "'))"
    elif typeClass == 'H5T_OPAQUE':
        if shape:
            raise TypeError(
                "Opaque Type is not supported for variable len types")
        if 'size' not in typeItem:
            raise KeyError("'size' not provided")
        nSize = int(typeItem['size'])
        if nSize <= 0:
            raise TypeError("'size' must be non-negative")
        code += "'V{}'".format(nSize)
    elif typeClass == 'H5T_ARRAY':
        if not shape:
            raise KeyError("'shape' must be provided for array types")
        if 'base' not in typeItem:
            raise KeyError("'base' not provided")
        baseType = getNumpyTypename(typeItem['base']['base'])
        if type(baseType) not in (str, unicode):
            raise TypeError(
                "Array type is only supported for predefined base types")
        # should be one of the predefined types
        code += "'{}{}'".format(shape, baseType)
    elif typeClass == 'H5T_COMPOUND':
        if 'fields' not in typeItem:
            raise KeyError("'fields' must be provided for compound types")
        if type(typeItem['fields']) is not list:
            raise TypeError("compound 'fields' value must be a list")
        dt_arg = list()
        for fld in typeItem['fields']:
            dt_arg.append(
                "('{}', {})".format(fld['name'], _dtype(fld['type'])))
        code = '[' + ', '.join(dt_arg) + ']'
    else:
        raise TypeError("%s: Invalid type class" % typeClass)

    return code


def valueToString(attr_json):
    value = attr_json["value"]
    return json.dumps(value)


def doAttribute(attr_json, parent_var):
    print(parent_var + ".attrs['" + attr_json['name'] + "'] = "
           + valueToString(attr_json))


def getObjectName(obj_json):
    name = "???"
    if "alias" in obj_json:
        alias = obj_json["alias"]
        if type(alias) in (list, tuple):
            if len(alias) > 0:
                name = alias[0]
        else:
            name = alias
    return name


def doAttributes(obj_json, parent_var, is_dimscale=False, is_dimension=False):
    if len(obj_json.get('attributes', [])) == 0:
        return
    attrs_json = obj_json["attributes"]

    print("# creating attributes for '" + getObjectName(obj_json) + "'")
    for attr_json in attrs_json:
        if is_dimscale and attr_json['name'] in ('CLASS', 'REFERENCE_LIST',
                                                 'NAME'):
                continue
        if is_dimension and attr_json['name'] == 'DIMENSION_LIST':
            continue
        doAttribute(attr_json, parent_var)


def getObjectVariableName(title):
    char_list = list(title)
    for i in range(len(char_list)):
        ch = char_list[i]
        if (ch >= 'A' and ch <= 'z') or (ch >= '0' and ch <= '9'):
            pass  # ok char
        else:
            char_list[i] = '_'  # replace with underscore
    var_name = "".join(char_list)
    if char_list[0] >= '0' and char_list[0] <= '9':
        var_name = 'v' + var_name   # pre-pend with a non-numeric
    if var_name in variable_names:
        for i in range(1, 99):
            if (var_name + '_' + str(i)) not in variable_names:
                var_name += '_' + str(i)
                break
        else:
            raise ValueError(
                'Unable to construct Python variable name for: "%s"' % title)
    variable_names.add(var_name)  # add to our list of variable names

    return var_name


def doGroup(h5json, group_id, group_name, parent_var):
    groups = h5json["groups"]
    group_json = groups[group_id]
    print("\n\n# group -- ", group_json['alias'][0])
    group_var = getObjectVariableName(group_name)
    print("{0} = {1}.create_group('{2}')".format(
        group_var, parent_var, group_name))
    doAttributes(group_json, group_var)
    doLinks(h5json, group_json, group_var)


def _dims2str(dims, kwd=''):
    """Convert a list of integers to a string representing a dimension tuple.

    :arg list dims: A dimension list.
    :arg str kwd: Optional keyword string.
    """
    dims = [str(d) if d != 'H5S_UNLIMITED' else 'None' for d in dims]
    if len(dims) == 1:
        # Produce correct tuple when dim rank is 1...
        dims.append('')
    if kwd:
        kwd += '='
    return ', {}({})'.format(kwd, ','.join(dims))


def doDataset(h5json, dset_id, dset_name, parent_var):
    datasets = h5json["datasets"]
    dset_json = datasets[dset_id]
    print("\n# make dataset: ", dset_json['alias'][0])
    dset_var = getObjectVariableName(dset_name)
    dtLine = getBaseDataType(dset_json["type"])  # "dt = ..."
    print(dtLine)

    shape = ''
    maxshape = ''
    chunks = ''
    flt = ''
    cp = dset_json.get('creationProperties', {})
    if "shape" in dset_json:
        shape_json = dset_json["shape"]
        if shape_json["class"] == "H5S_SIMPLE":
            shape = _dims2str(shape_json["dims"])

            if 'maxdims' in shape_json:
                maxshape = _dims2str(shape_json['maxdims'], kwd='maxshape')

            layout = cp.get('layout', {}).get('class', 'H5D_CONTIGUOUS')
            if layout == 'H5D_CHUNKED':
                chunks = _dims2str(cp['layout']['dims'], kwd='chunks')

                filters = cp.get('filters', [])
                for f in filters:
                    if f['class'] == 'H5Z_FILTER_DEFLATE':
                        flt += (", compression='gzip', "
                                "compression_opts={:d}").format(f['level'])
                    elif f['class'] == 'H5Z_FILTER_FLETCHER32':
                        flt += ', fletcher32=True'
                    elif f['class'] == 'H5Z_FILTER_SHUFFLE':
                        flt += ', shuffle=True'
                    elif f['class'] == 'H5Z_FILTER_SCALEOFFSET':
                        flt += ', scaleoffset={:d}'.format(f['scaleOffset'])
                    else:
                        raise NotImplementedError(
                            '%s: Filter not supported yet' % f['class'])

        elif shape_json['class'] == 'H5S_SCALAR':
            shape = ', ()'

    if 'fillValue' in cp:
        fv = ', fillvalue={}'.format(cp['fillValue'])
    else:
        fv = ''

    code_line = ("{} = {}.create_dataset('{}'{}{}{}{}{}"
                 ", dtype=dt)").format(dset_var, parent_var, dset_name, shape,
                                       maxshape, chunks, fv, flt)
    print(code_line)
    print("# initialize dataset values here")

    dscale = _is_dimscale(dset_json.get('attributes', []))
    if dscale:
        # Find out dimension scale's name, if available...
        for a in dset_json.get('attributes', []):
            if a['name'] == 'NAME':
                ds_name = a['value']
                break
        else:
            ds_name = None
        dimscales.update({dset_id: ds_name})
    dim = _is_dimlist(dset_json.get('attributes', []))
    if dim:
        dimensions.append(dset_id)

    doAttributes(dset_json, dset_var, is_dimscale=dscale, is_dimension=dim)


def doLink(h5json, link_json, parent_var):
    if link_json["class"] == "H5L_TYPE_EXTERNAL":
        print("{0}['{1}'] = h5py.ExternalLink('{2}', '{3}')".format(parent_var, link_json["title"], link_json["file"], link_json["h5path"]))
    elif link_json["class"] == "H5L_TYPE_SOFT":
        print("{0}['{1}'] = h5py.SoftLink('{2}')".format(
            parent_var, link_json["title"], link_json["h5path"]))
    elif link_json["class"] == "H5L_TYPE_HARD":
        if link_json["collection"] == "groups":
            doGroup(h5json, link_json["id"], link_json["title"], parent_var)
        elif link_json["collection"] == "datasets":
            doDataset(h5json, link_json["id"], link_json["title"], parent_var)
        elif link_json["collection"] == "datatypes":
            pass  # todo
        else:
            raise Exception("unexpected collection name: "
                            + link_json["collection"])
    elif link_json["class"] == "H5L_TYPE_UDLINK":
        print("# ignoring user defined link: '{0}'".format(link_json["title"]))
    else:
        raise Exception("unexpected link type: " + link_json["class"])


def doLinks(h5json, group_json, parent_var):
    links = group_json["links"]
    for link in links:
        doLink(h5json, link, parent_var)


def _is_dimscale(attrs):
    """Check if the dataset is a dimension scale.

    :arg list attrs: All dataset's attributes.
    """
    # Check if REFERENCE_LIST attribute is present...
    ref_list = any(a['name'] == 'REFERENCE_LIST'
                   and a['type']['class'] == 'H5T_COMPOUND'
                   for a in attrs)

    # Check if CLASS attribute is present...
    cls_ = any(a['name'] == 'CLASS'and a['value'] == 'DIMENSION_SCALE'
               for a in attrs)

    if ref_list and cls_:
        return True
    else:
        return False


def _is_dimlist(attrs):
    """Check if the dataset has dimension scales attached.

    :arg list attrs: All dataset's attributes.
    """
    # Check if DIMENSION_LIST attribute is present...
    dim_list = any(a['name'] == 'DIMENSION_LIST'
                   and a['type']['class'] == 'H5T_VLEN'
                   for a in attrs)

    return True if dim_list else False


def _dset_paths(h5json):
    grps = [{'id': h5json['root'], 'path': '/'}]
    dsets = {}

    def _get_hard_links(grpjson, collection):
        links = list()
        for l in grpjson.get('links', []):
            if l['class'] == 'H5L_TYPE_HARD' and l['collection'] == collection:
                links.append(l)
        return links

    def tree_walker(ginfo):
        dlinks = _get_hard_links(h5json['groups'][ginfo['id']], 'datasets')
        for dl in dlinks:
            dsets.update({dl['id']: pp.join(ginfo['path'], dl['title'])})

        glinks = _get_hard_links(h5json['groups'][ginfo['id']], 'groups')
        chld_grps = list()
        for gl in glinks:
            chld_grps.append({'id': gl['id'],
                              'path': pp.join(ginfo['path'], gl['title'])})
        grps.extend(chld_grps)
        for cg in chld_grps:
            tree_walker(cg)

    tree_walker(grps[0])

    return dsets


def doDimensions(h5json, dimensions, dimscales, parent_var):
    if len(dimensions) == 0:
        return

    # Generate HDF5 paths for all datasets...
    dset_path = _dset_paths(h5json)

    print('\n\n'
          '#\n'
          '# Adding dimensions\n'
          '#\n'
          '\n')

    print('# Creating dimension scales')
    for dsid, name in dimscales.iteritems():
        if dimscales[dsid]:
            name = ", '{}'". format(dimscales[dsid])
        else:
            name = ''
        print("h5py.h5ds.set_scale({}['{}'].id{})".format(
            parent_var, dset_path[dsid], name))

    print('\n# Attaching dimension scales to their datasets')
    for dsid in dimensions:
        dsid_path = dset_path[dsid]
        print("\n# Dataset: {}".format(dsid_path))
        for attr in h5json['datasets'][dsid].get('attributes', []):
            if attr['name'] == 'DIMENSION_LIST':
                dim_list = attr['value']
                break
        else:
            raise ValueError('%s: DIMENSION_LIST attribute not found'
                             % dsid_path)

        for idx, ds in enumerate(dim_list):
            for d in ds:
                did = pp.split(d)[-1]
                did_path = dset_path[did]
                print("{}['{}'].dims[{:d}].attach_scale({}['{}'])".format(
                    parent_var, dsid_path, idx, parent_var, did_path))


def main():
    parser = argparse.ArgumentParser(
        usage='%(prog)s [-h] <json_file> <out_filename>')
    parser.add_argument('in_filename', nargs='+',
                        help='JSon file to be converted to h5py')
    parser.add_argument('out_filename', nargs='+',
                        help='name of HDF5 file to be created by generated code')
    args = parser.parse_args()

    text = open(args.in_filename[0]).read()

    # parse the json file
    h5json = json.loads(text)

    if "root" not in h5json:
        raise Exception("no root key in input file")
    root_uuid = h5json["root"]

    filename = args.out_filename[0]
    file_variable = 'f'

    print("import h5py")
    print("import numpy as np")
    print(" ")
    print("print 'creating file: {0}'".format(filename))
    print("{0} = h5py.File('{1}', 'w')".format(file_variable, filename))
    print(" ")

    group_json = h5json["groups"]
    root_json = group_json[root_uuid]
    doAttributes(root_json, file_variable)
    doLinks(h5json, root_json, file_variable)
    doDimensions(h5json, dimensions, dimscales, file_variable)

    print("\n\nprint('done!')")

main()
