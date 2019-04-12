/*
 Navicat Premium Data Transfer

 Source Server         : 西丽云滁州环保数据库（测试环境）
 Source Server Type    : MySQL
 Source Server Version : 50720
 Source Host           : 14.116.179.252
 Source Database       : enp

 Target Server Type    : MySQL
 Target Server Version : 50720
 File Encoding         : utf-8

 Date: 12/19/2018 17:13:22 PM
*/

SET NAMES utf8;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
--  Table structure for `forecast_city_list`
-- ----------------------------
DROP TABLE IF EXISTS `forecast_city_list`;
CREATE TABLE `forecast_city_list` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `province_id` int(11) DEFAULT NULL,
  `city_id` int(11) DEFAULT NULL,
  `city_name` varchar(20) DEFAULT NULL,
  `longitude` decimal(10,0) DEFAULT NULL,
  `latitude` decimal(10,0) DEFAULT NULL,
  PRIMARY KEY (`id`) USING BTREE
) ENGINE=InnoDB AUTO_INCREMENT=16 DEFAULT CHARSET=utf8 ROW_FORMAT=DYNAMIC;

-- ----------------------------
--  Records of `forecast_city_list`
-- ----------------------------
BEGIN;
INSERT INTO `forecast_city_list` VALUES ('1', null, '341300', '宿州市                 ', '117', '34'), ('2', null, '340600', '淮北市                 ', '117', '34'), ('3', null, '341600', '亳州市                 ', '116', '34'), ('4', null, '341200', '阜阳市                 ', '116', '33'), ('5', null, '340300', '蚌埠市                 ', '117', '33'), ('6', null, '340400', '淮南市                 ', '117', '33'), ('7', null, '341100', '滁州市                 ', '118', '32'), ('8', null, '341500', '六安市                 ', '116', '32'), ('9', null, '340500', '马鞍山市                ', '119', '32'), ('10', null, '340800', '安庆市                 ', '117', '31'), ('11', null, '340200', '芜湖市                 ', '118', '31'), ('12', null, '340700', '铜陵市                 ', '118', '31'), ('13', null, '341800', '宣城市                 ', '119', '31'), ('14', null, '341700', '池州市                 ', '117', '31'), ('15', null, '341000', '黄山市                 ', '118', '30'), ('16', null, '340100', '合肥市                 ', '117', '32');
COMMIT;

SET FOREIGN_KEY_CHECKS = 1;
