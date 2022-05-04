from pyspark import SparkContext
import time
import json
import os
import statistics
import ast


def helper_mapvalues_std(value):
    if len(value) == 1:
        return 0
    else:
        return statistics.stdev(value)


def helper_mapValues_handle_leftouterjoin(value):
    if value[1] is None:
        return value[0]+[0]
    else:
        return value[0]+[int(value[1])]


def filter_persist_review(input_dir, output_dir, spark_context):
    start_time = time.time()
    review_path = os.path.join(input_dir, 'yelp_academic_dataset_review.json')
    output_path = os.path.join(output_dir, 'review_2018.json')
    raw_data = spark_context.textFile(review_path).map(lambda x: json.loads(x))
    reviews = raw_data.filter(lambda x: str(x['date']).startswith('2018')).map(
        lambda x: {
            "review_id": x['review_id'],
            "user_id": x["user_id"],
            "business_id": x["business_id"],
            "stars": x["stars"],
            "date": x["date"],
            "useful": x["useful"],
            "funny": x["funny"],
            "cool": x["cool"]})
    with open(output_path, 'w') as fhand:
        for review in reviews.collect():
            fhand.write(json.dumps(review)+'\n')
    print('Collapsed: ', time.time()-start_time)
    pass


def helper_map_get_feat_from_business(iterator):
    attributes = iterator['attributes']
    if attributes is None:
        restaurants_take_out = 0
        business_parking = 0
    else:
        restaurants_take_out = attributes.get('RestaurantsTakeOut')
        if restaurants_take_out is None:
            restaurants_take_out = 0
        else:
            restaurants_take_out = int(bool(restaurants_take_out))

        business_parking = attributes.get('BusinessParking')
        if business_parking is None:
            business_parking = 0
        else:
            business_parking = ast.literal_eval(business_parking)
            # Handle the case 'None'.
            if business_parking is None:
                business_parking = 0
            else:
                business_parking = [val for val in business_parking.values() if val is not None]
                # Handle the case when all values are None.
                if len(business_parking) > 0:
                    business_parking = int(max(business_parking))
                else:
                    business_parking = 0

    return iterator['business_id'], (iterator['categories'],
                                     iterator['review_count'],
                                     restaurants_take_out,
                                     business_parking,
                                     iterator['stars'])


def persist_review_join_business(input_dir, output_dir, spark_context, bias=1e-1):
    start_time = time.time()

    filtered_review_path = os.path.join(output_dir, 'review_2018.json')
    business_path = os.path.join(input_dir, 'yelp_academic_dataset_business.json')
    photo_cnt_path = os.path.join(output_dir, 'photo_cnt.csv')
    tip_cnt_path = os.path.join(output_dir, 'tip_cnt.csv')

    output_path = os.path.join(output_dir, 'all_features_2018_v2.csv')

    business_data = spark_context.textFile(business_path).map(lambda x: json.loads(x))

    business_feats = business_data.map(helper_map_get_feat_from_business)

    filtered_reviews = spark_context.textFile(filtered_review_path).map(lambda x: json.loads(x))

    # The bias prevents the number of cool, useful and funny to be zero.
    cool_weighted = filtered_reviews.map(
        lambda x: (x['business_id'], ((int(x['cool'])+bias)*int(x['stars']), bias+int(x['cool'])))).reduceByKey(
        lambda x, y: (x[0]+y[0], x[1]+y[1])).mapValues(lambda value: value[0]/value[1])

    funny_weighted = filtered_reviews.map(
        lambda x: (x['business_id'], ((int(x['funny'])+bias)*int(x['stars']), bias+int(x['funny'])))).reduceByKey(
        lambda x, y: (x[0]+y[0], x[1]+y[1])).mapValues(lambda value: value[0]/value[1])

    useful_weighted = filtered_reviews.map(
        lambda x: (x['business_id'], ((int(x['useful'])+bias)*int(x['stars']), bias+int(x['useful'])))).reduceByKey(
        lambda x, y: (x[0]+y[0], x[1]+y[1])).mapValues(lambda value: value[0]/value[1])

    photo_cnt = spark_context.textFile(photo_cnt_path)
    photo_cnt_header = photo_cnt.first()
    photo_cnt = photo_cnt.filter(lambda x: x != photo_cnt_header).map(
        lambda x: (x.split(',')[1], x.split(',')[2]))

    tip_cnt = spark_context.textFile(tip_cnt_path)
    tip_cnt_header = tip_cnt.first()
    tip_cnt = tip_cnt.filter(lambda x: x != tip_cnt_header).map(
        lambda x: (x.split(',')[1], x.split(',')[2]))

    review_stars = filtered_reviews.map(lambda x: (x['business_id'], int(x['stars']))).groupByKey()
    review_stars_median = review_stars.mapValues(lambda x: statistics.median(x))
    review_stars_std = review_stars.mapValues(helper_mapvalues_std)

    # Join the data following the sequence:
    # median stars, std stars, useful, funny, cool, tip count, business feats.
    joined_data = review_stars_median.join(review_stars_std).join(useful_weighted).mapValues(
        lambda x: list(x[0])+[x[1]]).join(funny_weighted).mapValues(
        lambda x: x[0]+[x[1]]).join(cool_weighted).mapValues(
        lambda x: x[0]+[x[1]]).leftOuterJoin(tip_cnt).mapValues(
        helper_mapValues_handle_leftouterjoin).join(business_feats).mapValues(lambda x: x[0] + list(x[1]))
    print(joined_data.collect())

    with open(output_path, 'w') as fhand:
        fhand.write('business_id,' +
                    'review_stars_median,' +
                    'review_stars_std,' +
                    'weighted_useful_stars,' +
                    'weighted_funny_stars,' +
                    'weighted_cool_stars,' +
                    'tip_cnt,' +

                    'categories,' +
                    'review_count,' +
                    'restaurants_take_out,' +
                    'business_parking,' +
                    'stars\n')
        for row in joined_data.collect():
            fhand.write(str(row[0])+',')
            feat_len = len(row[1])
            for i in range(feat_len):
                feat = str(row[1][i])
                if i == feat_len - 5:
                    feat = '"' + str(row[1][i]) + '"'
                fhand.write(feat)
                if i != feat_len - 1:
                    fhand.write(',')
            fhand.write('\n')
    print('Collapsed: ', time.time()-start_time)
    pass


if __name__ == "__main__":
    input_folder = '/Volumes/Ark/yelp_dataset/'
    output_folder = '/Volumes/Ark/yelp_dataset/generated/'
    sc = SparkContext()
    sc.setLogLevel('ERROR')

    # filter_persist_review(input_folder, output_folder, sc)

    persist_review_join_business(input_folder, output_folder, sc)
