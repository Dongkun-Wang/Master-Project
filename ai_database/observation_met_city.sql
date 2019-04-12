/*
database: ecoai
*/

SET NAMES utf8;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
--  Table structure for `observation_met_city`
-- ----------------------------
DROP TABLE IF EXISTS `observation_met_city`;
CREATE TABLE `observation_met_city` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `city_id` int(11) DEFAULT NULL,
  `file_date` datetime DEFAULT NULL,
  `met_type` varchar(16) DEFAULT NULL,
  `met_unit` varchar(8) DEFAULT NULL,
  `observed_time` datetime DEFAULT NULL,
  `met_value` varchar(16) DEFAULT NULL, 
  PRIMARY KEY (`id`) USING BTREE
) ENGINE=InnoDB AUTO_INCREMENT=1 DEFAULT CHARSET=utf8 ROW_FORMAT=DYNAMIC COMMENT='城市气象观测表';
