# clifter_slam.config

[source file clifter_slam.config](https://github.com/slowy07/clifter_slam/blob/main/clifter_slam/config/cfgnode.py)

``CfgNode`` is a node int he configuration tree. it's a simple wrapper around a dict and supports access to attributes via keys.

```python
clone()
# recusively copy this CfgNode
```

```python
defrost()
# make this CfgNode and all of its childern mutable
```

```python
dump(**kwargs)
# dump CfgNode to a string
```

```python
freeze()
# make this CfgNode and all of its chidern immutable
```

```python
is_frozen()
# return immutability
```

```python
key_is_deprecated(full_key: str)
# test if a key is renamed
```

```python
key_is_renamed(full_key: str)
# test if a key is renamed
```

```python
classmethod load_cfg(cfg_file_obj_or_str)
# load a configuration into CfgNode
```
parameters:

**cfg_file_obj_or_str** a file object backed by a YAML file. file object backed by a python source file that exports an sttribute ``cfg`` (dict or cfgNode). string that can be parsed as valid YAML

```python
merge_from_file(cfg_filename: str)
# load a yaml config file and merge it with this CfgNode
```
parameters:

**cfg_filename** is config file path

```python
marge_from_list(cfg_list: list)
# merge config(ket, values) in a list (eg. from commandline) into this CfgNode
```

```python
merge_from_ther_cfg(cfg_other)
# merge cfg other into the current CfgNode
```

```python
register_deprecated_key(key: str)
# register key (eg. CLIFTEEER) a deprecated option. when merging deprecated keys, a warning is generated and the key is ignored
```

```python
register_renamed_key(old_name: str, new_name: str, message: Optional[str] = None)
```
register a key as a=having been renamed from ``old_name`` to ``new_name``. when mergin a renamed key, an exception is throw alerting th user to the fact that the kay has been renamed
