"""For now I am compiling examples of unexpected issues in the data"""
import datetime
import os
import pickle

import pandas

import pandas_utils
import unclassified, resources
from mxpnl import accessors, tools
from mxpnl.accessors import transformed_event_df, bare_event, reg_users
from mxpnl.tools import pandas_timestamp_to_datetime
from unclassified import ContextTimer


class NoiseExamples:

    @classmethod
    def events_before_user_created(cls):
        """User Created is not always near the beginning of a distinct_id history. This function shows some counts of
        various patterns around User Created, Media consumption and subscriptions"""
        print("estimated run time if data is cached: <60s")
        dfu = transformed_event_df('User Created', 20190201, 20200215, transform=[bare_event, reg_users]).sort_values(
            'time')
        dff = dfu.groupby('distinct_id').head(1).rename(columns={'time': 'created_at'}).drop('event_type', axis=1)

        dff_keep_path = fr'{resources.PERSONAS}\dff_keep.pik'
        dff_keep = unclassified.pickle_load(dff_keep_path, safe=False)
        dff_unknown_path = fr'{resources.PERSONAS}\dff_unknown.pik'
        dff_unknown = unclassified.pickle_load(dff_unknown_path, safe=False)
        if dff_keep is None or dff_unknown is None:
            dff_keep = None
            dff_unknown = dff
            MEDIA = {'Media Start', 'Media Complete'}

            def log(msg):
                print("{:%H:%M:%S}: {}".format(datetime.datetime.now(), msg))

            for dt in unclassified.date_iterator(20190201, 20200215):
                log("{:%Y%m%d}".format(dt))
                dfm = transformed_event_df(MEDIA, dt, transform=[bare_event])
                x = pandas.merge(dff_unknown, dfm['distinct_id time'.split()], how='left', on='distinct_id').groupby(
                    "distinct_id created_at".split(), as_index=False).min()
                y = x.query("created_at < time").drop('time', axis=1)
                if dff_keep is None:
                    dff_keep = y
                else:
                    dff_keep = pandas.concat([dff_keep, y])
                dff_unknown = x[x.time.isnull()].drop('time', axis=1)
            os.makedirs(resources.PERSONAS, exist_ok=True)
            with open(dff_keep_path, 'wb') as fh:
                pickle.dump(dff_keep, fh)
            with open(dff_unknown_path, 'wb') as fh:
                pickle.dump(dff_unknown, fh)
        x = pandas.concat(
            [dff.assign(legacy=1).set_index('distinct_id created_at'.split()),
             dff_keep.assign(active=1).set_index('distinct_id created_at'.split()),
             dff_unknown.assign(inactive=1).set_index('distinct_id created_at'.split()),
             ], sort=False, axis=1).reset_index()
        x['legacy'] = x.legacy.where(x.active.isnull() & x.inactive.isnull())
        dfpers = pandas.melt(x, id_vars='distinct_id created_at'.split(), var_name='uc_type')
        dfpers = dfpers.dropna().drop('value', axis=1)
        dfpers['uc_type'] = pandas.Categorical(dfpers.uc_type)
        PURCHASE_EVENTS = set.union({'Admin Subscription Grant', 'Purchase', 'purchase', 'af_purchase', '$ae_iap'},
                                    tools.PURCHASE_EVENTS)
        dfp = transformed_event_df(PURCHASE_EVENTS, 20190201, 20200215, transform=[bare_event])
        dfp = pandas.merge(dfp, dfpers.query("uc_type == 'active'"), how='inner', on='distinct_id')
        # previously active users. These are user who had subscribed some time before the recent user created event but had't listened to media since 2019-02-01
        dfpa = pandas_timestamp_to_datetime(
            dfp.assign(days=(dfp.created_at - dfp.time) / (24 * 60 * 60)).sort_values('time').groupby(
                'distinct_id').head(1).sort_values('days'), 'time created_at'.split())
        dfpers = dfpers.assign(
            uc_type=dfpers.uc_type.where(~dfpers.distinct_id.isin(set(dfpa.query("days >= 10").distinct_id)),
                                         'prev_subs'))
        # and there are the "prepaid" users. These users probably responded to an email campaign that led to a subscription and then recreate/update the profile.
        dfpers = dfpers.assign(uc_type=dfpers.uc_type.where(
            ~dfpers.distinct_id.isin(set(dfpa.query("0 <= days and days < 10").distinct_id)), 'prepaid'))
        print("counts of various types of users created from 20190201 to 20200215")
        print("active = User Created event appears to be the first one for this distinct_id AND some media consumed")
        print("inactive is new User Created but no media consumed")
        print("legacy is distinct_id with no User Created before the first Media event. These users were probably\n"
              "  created before 20190201.")
        print("prev_subs are distinct_ids with some subscription event more than 10 days before the User Created.\n"
              "  They are legacy users who haven't been active with media but have some activity indicating previous\n"
              "  subscription")
        print("prepaid are distinct_ids that have no media before User Created, but they do have subscription shortly\n"
              "  before User Created. They maybe subscribe without trying the app, have a share or admin grant, or\n"
              "  maybe just noise.")
        print(dfpers.groupby('uc_type').count()['distinct_id'])

    @classmethod
    def content_view_not_firing(cls, study_date=20200217):
        """It appears that the intent is to fire a 'content view' event every time 'Media Compete' fires. The event
        does not always fire, and the misfire rate is significant depending on the build and os. This counts the
        number of times a Media Complete fires without an accompanying 'content view', on 20200217, broken out by
        build number and os.
        """
        print("should run in <30s if data is cached")
        def nearby_events(dfx):
            return dfx.assign(
                prev_event=dfx.event_type.shift().where(dfx.distinct_id == dfx.distinct_id.shift()),
                prev_delay=(dfx.time - dfx.time.shift()).where(dfx.distinct_id == dfx.distinct_id.shift()),
                next_event=dfx.event_type.shift(-1).where(dfx.distinct_id == dfx.distinct_id.shift(-1)),
                next_delay=(dfx.time.shift(-1) - dfx.time).where(dfx.distinct_id == dfx.distinct_id.shift(-1)),
            )

        def has_event_nearby(x, evt, near_evt, max_delay=600):
            return x.assign(near=(((x.prev_event == near_evt) & (x.prev_delay < max_delay)) | (
                        (x.next_event == near_evt) & (x.next_delay < max_delay))).values).query(
                "event_type == {!r}".format(evt))

        evt = 'Media Complete'
        near_evt = 'content view'

        dfm = tools.events_to_df([e for e in accessors.load_all_events(study_date) if
                                  e['event'] == 'Media Complete' or e['event'] == 'content view'])
        dfm = dfm.sort_values('distinct_id time'.split())
        dfm = nearby_events(dfm)
        dfm.query("distinct_id == '1245232'").head()
        dfm = dfm.assign(has_real_id=dfm.distinct_id.str.len() < 15)
        x = has_event_nearby(dfm, evt, near_evt)
        x = x.assign(track_id=x.trackId.fillna(-999).astype('float64'))
        # x.groupby('track_id')['near'].agg('mean count'.split()).sort_values('mean', ascending=False).sort_index()
        # x.query("track_id == 114").sort_values("near").head(10).T
        # x.groupby("mp_lib")['near'].agg('mean count'.split())
        # x.assign(near_delay=x.prev_delay.where(x.prev_event == near_evt, x.next_delay)).query(
        #     'event_type == {!r} and near == True'.format(evt)).groupby("mp_lib")['near_delay'].describe()
        x = x.assign(app_build_number=x['$app_build_number'].astype('float64')).query("app_build_number>=100")
        x = x.assign(near_delay=x.prev_delay.where(x.prev_event == near_evt, x.next_delay))
        print("Counts of Media Complete events having 'content view' nearby or not, broken down by OS and build.\n"
              "  False means the 'content view' did not fire. It appears that android has a problem")
        print(
            x.groupby(
                "$os app_build_number near".split())['time'].count().unstack('$os near'.split()))

        print()
        print("I suspect the misfiring of 'content view' on Android is due to the apparent lag between\n"
              "'Media Complete' and 'content view', as shown below. The lag is 2s on android more than half the time")
        print(x.query(f'event_type == {evt!r} and near == True').groupby("$os app_build_number".split())['near_delay'].describe().to_string())

        return x

    @classmethod
    def multiple_distinct_ids_one_subscription(cls):
        """The notion of distinct_id is not air-tight. Here are examples of multiple distinct_ids that fire
        subscription events with the same subscription_id. These are probably the same user, maybe using different
        devices to fire the event"""
        print("estimated run time if data is cached: <30s")
        with ContextTimer('loading events', verbose=True):
            z = accessors.transformed_event_df({'New Subscription', 'Renew Subscription', 'Resubscription'}, 20190201,
                                               20200315, transform=lambda df: df[
                    [c for c in df if c in {'event_type', 'distinct_id', 'subscription_id', 'time'}]])
        with ContextTimer("counting duplicate distinct_ids", verbose=True):
            z = pandas_utils.group_query(
                z.groupby('subscription_id distinct_id'.split()).head(1),
                'subscription_id',
                'count',
                'distinct_id > 1'
            )
        z['subscription_id'] = pandas.Categorical(z['subscription_id'])
        z = z.query("subscription_id != 'ADMIN' and subscription_id != 'EFL'")
        z = z.assign(subscription_id_num=z.subscription_id.cat.codes)
        z = pandas_utils.add_rank(z, 'subscription_id_num')
        print("Here are some examples of subscription_ids shared by multiple distinct_ids:")
        print(z.head(100).to_string())

        print()
        print("Some interesting subscription_ids (with large numbers of distinct_ids sharing subscription id:")
        print(z.sort_values('rank').groupby('subscription_id_num').tail(1).query('rank > 5').to_string())
        return z
