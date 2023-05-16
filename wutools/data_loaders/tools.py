import gzip
import hashlib
import os
import pickle
import sqlalchemy
import sqlparse
import urllib

ATHENA_CONN_STR = "awsathena+rest://{aws_access_key_id}:{aws_secret_access_key}@athena.{region_name}.amazonaws.com:443/"\
           "{schema_name}?s3_staging_dir={s3_staging_dir}"
def get_athena_engine(cred_dir):
    """cred_dir is the path to the directory holding your aws credentials file
    In collab this is usually "/content/drive/My Drive/aws"

    Returns:
        a SQL engine
    """
    with open(os.path.join(cred_dir, 'credentials')) as f:
        d = (dict(l.strip().split('=') for l in f.readlines() if '=' in l))
        engine = sqlalchemy.engine.create_engine(ATHENA_CONN_STR.format(
            aws_access_key_id=urllib.parse.quote_plus(d['aws_access_key_id']),
            aws_secret_access_key=urllib.parse.quote_plus(d['aws_secret_access_key']),
            region_name = 'us-west-2',
            work_group = 'primary',
            s3_staging_dir=urllib.parse.quote_plus('s3://llk-athena-results/amplitude/'),
            schema_name='llk_amplitude'))
    return engine


def hash_sql(sql):
    parsed = sqlparse.parse(sql)
    md5_hash = hashlib.md5()
    for t in (t.value for t in parsed[0].flatten() if not t.is_whitespace and not re.match(r'Token.Comment.*', str(t.ttype))):
        md5_hash.update(t.encode())
    return md5_hash.hexdigest()


def get_result_path(query, store_dir, tag=None, ):
    h = hash_sql(query)
    fname = (h if tag is None else f"{h}.{tag}") + '.pik.gz'
    return os.path.join(store_dir, fname)


def get_event_data(query, conn, store_dir=None, tag=None):

    if store_dir:
        fname = get_result_path(query, tag=tag, store_dir=store_dir)
        try:
            with gzip.open(fname, 'rb') as f:
                print(f"reading from {fname}")
                return pickle.load(f)
        except:
            print(f"could not load {fname}\nloading from db")

    with conn.cursor() as cursor:
        result = cursor.execute(query)
        result_set = result.fetchall()
        column_names = [desc[0] for desc in cursor.description]
    dict_result_set = [dict(zip(column_names, row)) for row in result_set]
    if store_dir:
        fname = get_result_path(query, tag=tag, store_dir=store_dir)
        try:
            with gzip.open(fname, 'wb') as f:
                print(f"writing to {fname}")
                pickle.dump(dict_result_set, f)
        except Exception as e:
            print(f"could not write to {fname}: {e}")
    return dict_result_set
